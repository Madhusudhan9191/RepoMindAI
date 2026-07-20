import { request } from "./client";

export function runIndexing(repoPath = ".", clearExisting = true) {
  return request("/index", {
    method: "POST",
    body: JSON.stringify({ repo_path: repoPath, clear_existing: clearExisting }),
  });
}

export function clearIndex() {
  return request("/index", {
    method: "DELETE",
  });
}
