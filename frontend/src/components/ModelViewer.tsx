import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export type RenderMode = "color" | "clay" | "wireframe";

interface Props {
  url: string;
  mode: RenderMode;
  autoRotate: boolean;
  /** Called once with a square render of the model after it first loads (used for gallery thumbnails). */
  onSnapshot?: (dataUrl: string) => void;
}

const CLAY = new THREE.MeshStandardMaterial({ color: 0xb8b8bc, roughness: 0.75, metalness: 0 });
const WIRE = new THREE.MeshBasicMaterial({ color: 0xffd43b, wireframe: true, transparent: true, opacity: 0.35 });

/** Orbitable GLB preview: vertex colours, clay, or wireframe-over-clay. */
export function ModelViewer({ url, mode, autoRotate, onSnapshot }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const state = useRef<{
    meshes: THREE.Mesh[];
    originals: Map<THREE.Mesh, THREE.Material | THREE.Material[]>;
    wires: THREE.Mesh[];
    controls?: OrbitControls;
  }>({ meshes: [], originals: new Map(), wires: [] });
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
    const loop = () => {
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
        const wire = new THREE.Mesh(mesh.geometry, WIRE);
        wire.visible = false;
        mesh.add(wire);
        state.current.wires.push(wire);
      });
      applyMode(state.current, modeRef.current);

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
        applyMode(state.current, "color"); // thumbnails always show the coloured model
        snap.setClearColor(0x1d1d21, 1);
        snap.render(scene, snapCam);
        applyMode(state.current, modeRef.current);
        grid.visible = true;
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
      state.current = { meshes: [], originals: new Map(), wires: [] };
    };
  }, [url]);

  const modeRef = useRef(mode);
  modeRef.current = mode;
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
  for (const mesh of s.meshes) mesh.material = mode === "color" ? s.originals.get(mesh)! : CLAY;
  for (const wire of s.wires) wire.visible = mode === "wireframe";
}
