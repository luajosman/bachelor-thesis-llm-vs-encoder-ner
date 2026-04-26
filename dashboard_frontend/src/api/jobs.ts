import { apiClient } from "./client";
import type { ActionName, Job, JobLogsResponse } from "../types/api";

export async function createJob(action: ActionName): Promise<Job> {
  const { data } = await apiClient.post<Job>("/jobs", { action });
  return data;
}

export async function fetchJobs(): Promise<Job[]> {
  const { data } = await apiClient.get<Job[]>("/jobs");
  return data;
}

export async function fetchJob(jobId: string): Promise<Job> {
  const { data } = await apiClient.get<Job>(`/jobs/${jobId}`);
  return data;
}

export async function fetchJobLogs(jobId: string): Promise<JobLogsResponse> {
  const { data } = await apiClient.get<JobLogsResponse>(`/jobs/${jobId}/logs`);
  return data;
}

export async function cancelJob(jobId: string): Promise<Job> {
  const { data } = await apiClient.post<Job>(`/jobs/${jobId}/cancel`);
  return data;
}

export function buildJobStreamUrl(jobId: string): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/api/jobs/${jobId}/stream`;
}
