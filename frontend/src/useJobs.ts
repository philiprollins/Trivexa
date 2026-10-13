import { useEffect, useRef, useState } from "react";
import type { Asset, Job } from "./api";

type Msg =
  | { type: "jobs"; data: Job[] }
  | { type: "job"; data: Job }
  | { type: "progress"; data: Pick<Job, "id" | "status" | "step" | "total" | "preview"> };

/** Live job state pushed from the backend over a websocket, reconnecting if the server restarts. */
export function useJobs(onJobFinished: (job: Job, asset: Asset | null) => void) {
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [connected, setConnected] = useState(false);
  const onFinishedRef = useRef(onJobFinished);
  onFinishedRef.current = onJobFinished;

  useEffect(() => {
    let ws: WebSocket;
    let closed = false;
    let retry: ReturnType<typeof setTimeout>;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/api/ws`);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry = setTimeout(connect, 1500);
      };
      ws.onmessage = (ev) => {
        const msg: Msg = JSON.parse(ev.data);
        if (msg.type === "jobs") {
          setJobs(Object.fromEntries(msg.data.map((j) => [j.id, j])));
        } else if (msg.type === "job") {
          const job = msg.data;
          setJobs((prev) => ({ ...prev, [job.id]: job }));
          if (job.status === "done" || job.status === "error" || job.status === "cancelled") {
            onFinishedRef.current(job, job.asset);
          }
        } else if (msg.type === "progress") {
          const p = msg.data;
          setJobs((prev) => (prev[p.id] ? { ...prev, [p.id]: { ...prev[p.id], ...p } } : prev));
        }
      };
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      ws.close();
    };
  }, []);

  return { jobs, connected };
}
