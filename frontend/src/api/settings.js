import { request } from "./client";

export function getSettings() {
  return request("/settings", {
    method: "GET",
  });
}

export function updateSettings(settings) {
  return request("/settings", {
    method: "POST",
    body: JSON.stringify(settings),
  });
}

export function getFiles(repoPath = ".") {
  return request(`/files?repo_path=${encodeURIComponent(repoPath)}`, {
    method: "GET",
  });
}
