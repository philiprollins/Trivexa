import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export type RenderMode = "color" | "parts" | "clay" | "wireframe";

interface Props {
  url: string;
  mode: RenderMode;
  autoRotate: boolean;
  /** Called once with a square render of the model after it first loads (used for gallery thumbnails). */
  onSnapshot?: (dataUrl: string) => void;
  /** Rigged models: draw the bones over the mesh. */
  skeleton?: boolean;
  /** Rigged models: sway every bone to show how the skin weights deform the mesh. */
  pose?: boolean;
  /** Animated models: play the GLB's first animation in a loop (paused on its current frame when false). */
  play?: boolean;
  /** Animated models: hold the root's horizontal position so travelling motions stay in view. */
  inPlace?: boolean;
}

interface PoseBone {
  bone: THREE.Bone;
  rest: THREE.Quaternion;
  axis: THREE.Vector3; // local bending axis, across the bone
  phase: number;
}

const POSE_AMPLITUDE = 0.35; // radians
const POSE_SPEED = 2.2; // radians per second

const CLAY = new THREE.MeshStandardMaterial({ color: 0xb8b8bc, roughness: 0.75, metalness: 0 });
const WIRE = new THREE.MeshBasicMaterial({ color: 0xffd43b, wireframe: true, transparent: true, opacity: 0.35 });
// Segmented models: one flat colour per part (mesh), in node order. Distinct hues first, then lighter/darker repeats.
const PART_COLORS = [
  0xe6194b, 0x3cb44b, 0xffe119, 0x4363d8, 0xf58231, 0x911eb4, 0x46f0f0, 0xf032e6, 0xbcf60c, 0xfabebe,
  0x008080, 0xe6beff, 0x9a6324, 0xfffac8, 0x800000, 0xaaffc3, 0x808000, 0xffd8b1, 0x000075, 0x808080,
];
const PART_MATERIALS = PART_COLORS.map((color) => new THREE.MeshStandardMaterial({ color, roughness: 0.7, metalness: 0 }));

/** Orbitable GLB preview: vertex colours, clay, or wireframe-over-clay; segmented models can colour each part, rigged
 * models can show and pose their bones, animated ones play their clip. */
export function ModelViewer({ url, mode, autoRotate, onSnapshot, skeleton = false, pose = false, play = false, inPlace = false }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const state = useRef<{
    meshes: THREE.Mesh[];
    originals: Map<THREE.Mesh, THREE.Material | THREE.Material[]>;
    wires: THREE.Mesh[];
    controls?: OrbitControls;
    helper?: THREE.SkeletonHelper;
    bones: PoseBone[];
    mixer?: THREE.AnimationMixer;
    rootTrack?: { track: THREE.VectorKeyframeTrack; travel: Float32Array; held: Float32Array };
  }>({ meshes: [], originals: new Map(), wires: [], bones: [] });
  const snapshotRef = useRef(onSnapshot);
  snapshotRef.current = onSnapshot;

  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x404048, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.8);
    key.position.set(2, 3, 2.5);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xffffff, 0.6);
    rim.position.set(-2.5, 1.5, -2);
    scene.add(rim);
    const grid = new THREE.GridHelper(4, 20, 0x3a3a42, 0x26262c);
    scene.add(grid);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotateSpeed = 1.5;
    state.current.controls = controls;

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = el;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    let frame = 0;
    let posed = false;
    const clock = new THREE.Clock();
    const animClock = new THREE.Clock(); // its own: getElapsedTime() on the pose clock also advances its delta
    const q = new THREE.Quaternion();
    const loop = () => {
      const { bones, mixer } = state.current;
      const delta = animClock.getDelta();
      if (mixer && playRef.current) mixer.update(delta);
      if (poseRef.current && bones.length) {
        const t = clock.getElapsedTime() * POSE_SPEED;
        for (const b of bones) b.bone.quaternion.copy(b.rest).multiply(q.setFromAxisAngle(b.axis, POSE_AMPLITUDE * Math.sin(t + b.phase)));
        posed = true;
      } else if (posed) {
        for (const b of bones) b.bone.quaternion.copy(b.rest);
        posed = false;
      }
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(loop);
    };
    loop();

    let disposed = false;
    new GLTFLoader().load(url, (gltf) => {
      if (disposed) return;
      const root = gltf.scene;
      const box = new THREE.Box3().setFromObject(root);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      root.position.sub(center).add(new THREE.Vector3(0, size.y / 2, 0)); // stand on the grid
      scene.add(root);

      // Collect first: adding wireframe children while traversing would make traverse visit them too.
      const found: THREE.Mesh[] = [];
      root.traverse((obj) => {
        if ((obj as THREE.Mesh).isMesh) found.push(obj as THREE.Mesh);
      });
      found.forEach((mesh) => {
        mesh.geometry.computeVertexNormals();
        state.current.meshes.push(mesh);
        state.current.originals.set(mesh, mesh.material);
        // A skinned mesh's overlay must be skinned too, or it stays in the rest pose while the mesh moves.
        const skinned = (mesh as THREE.SkinnedMesh).isSkinnedMesh ? (mesh as THREE.SkinnedMesh) : null;
        let wire: THREE.Mesh;
        if (skinned) {
          const skinnedWire = new THREE.SkinnedMesh(mesh.geometry, WIRE);
          skinnedWire.bind(skinned.skeleton, skinned.bindMatrix);
          wire = skinnedWire;
        } else {
          wire = new THREE.Mesh(mesh.geometry, WIRE);
        }
        wire.visible = false;
        mesh.add(wire);
        state.current.wires.push(wire);
      });
      applyMode(state.current, modeRef.current);

      const skinned = found.find((m) => (m as THREE.SkinnedMesh).isSkinnedMesh) as THREE.SkinnedMesh | undefined;
      if (skinned) {
        const helper = new THREE.SkeletonHelper(root);
        (helper.material as THREE.LineBasicMaterial).linewidth = 2;
        helper.visible = skeletonRef.current;
        scene.add(helper);
        state.current.helper = helper;
        state.current.bones = skinned.skeleton.bones
          .filter((bone) => (bone.parent as THREE.Bone | null)?.isBone) // the root moves the whole model: leave it
          .map((bone, i) => {
            // Bend across the bone: perpendicular to the direction from its head to its first child (or, for a
            // leaf, from its parent to it).
            const child = bone.children.find((c) => (c as THREE.Bone).isBone);
            const dir = (child ? child.position : bone.position).clone().normalize();
            const axis = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0));
            if (axis.lengthSq() < 1e-4) axis.set(1, 0, 0);
            return { bone, rest: bone.quaternion.clone(), axis: axis.normalize(), phase: i * 0.6 };
          });
      }

      if (gltf.animations.length) {
        // Started now, but advanced only from the next frame: the thumbnail below still shows the rest pose.
        const mixer = new THREE.AnimationMixer(root);
        const clip = gltf.animations[0];
        mixer.clipAction(clip).play();
        state.current.mixer = mixer;
        // The root's translation track: keep the travelling values and a copy held at the first frame's X/Z.
        const track = clip.tracks.find((t) => t.name.endsWith(".position")) as THREE.VectorKeyframeTrack | undefined;
        if (track) {
          const travel = Float32Array.from(track.values);
          const held = Float32Array.from(travel);
          for (let i = 0; i < held.length; i += 3) {
            held[i] = travel[0];
            held[i + 2] = travel[2];
          }
          state.current.rootTrack = { track, travel, held };
          track.values.set(inPlaceRef.current ? held : travel);
        }
      }

      const radius = size.length() / 2;
      const target = new THREE.Vector3(0, size.y / 2, 0);
      controls.target.copy(target);
      camera.position.copy(target).add(new THREE.Vector3(0.6, 0.35, 1).normalize().multiplyScalar(radius * 3.2));
      camera.near = radius / 100;
      camera.far = radius * 100;
      camera.updateProjectionMatrix();
      controls.update();

      if (snapshotRef.current) {
        // Square, grid-less render for the gallery thumbnail.
        const snap = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
        snap.setSize(384, 384, false);
        snap.outputColorSpace = THREE.SRGBColorSpace;
        const snapCam = camera.clone();
        snapCam.aspect = 1;
        snapCam.updateProjectionMatrix();
        grid.visible = false;
        if (state.current.helper) state.current.helper.visible = false;
        applyMode(state.current, "color"); // thumbnails always show the coloured model
        snap.setClearColor(0x1d1d21, 1);
        snap.render(scene, snapCam);
        applyMode(state.current, modeRef.current);
        grid.visible = true;
        if (state.current.helper) state.current.helper.visible = skeletonRef.current;
        snapshotRef.current(snap.domElement.toDataURL("image/png"));
        snap.dispose();
      }
    });

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      state.current.helper?.dispose();
      state.current.mixer?.stopAllAction();
      state.current = { meshes: [], originals: new Map(), wires: [], bones: [] };
    };
  }, [url]);

  const modeRef = useRef(mode);
  modeRef.current = mode;
  const skeletonRef = useRef(skeleton);
  skeletonRef.current = skeleton;
  const poseRef = useRef(pose);
  poseRef.current = pose;
  const playRef = useRef(play);
  playRef.current = play;
  const inPlaceRef = useRef(inPlace);
  inPlaceRef.current = inPlace;
  useEffect(() => {
    const r = state.current.rootTrack;
    if (r) r.track.values.set(inPlace ? r.held : r.travel);
  }, [inPlace]);
  useEffect(() => {
    if (state.current.helper) state.current.helper.visible = skeleton;
  }, [skeleton]);
  useEffect(() => applyMode(state.current, mode), [mode]);
  useEffect(() => {
    if (state.current.controls) state.current.controls.autoRotate = autoRotate;
  }, [autoRotate, url]);

  return <div ref={host} className="model-viewer" />;
}

function applyMode(
  s: { meshes: THREE.Mesh[]; originals: Map<THREE.Mesh, THREE.Material | THREE.Material[]>; wires: THREE.Mesh[] },
  mode: RenderMode,
) {
  s.meshes.forEach((mesh, i) => {
    mesh.material = mode === "color" ? s.originals.get(mesh)! : mode === "parts" ? PART_MATERIALS[i % PART_MATERIALS.length] : CLAY;
  });
  for (const wire of s.wires) wire.visible = mode === "wireframe";
}
