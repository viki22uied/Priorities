import { fetchWithTimeout } from "@/lib/api";
import type { Category, Cluster, Urgency } from "@/lib/types";

export interface ApiSubmission {
  id: string;
  topic: string;
  description: string;
  issueType: string;
  severity: string;
  locality: string;
  submittedFor: string;
  aiTags?: string[];
  latitude?: number | null;
  longitude?: number | null;
  hasImage?: boolean;
  createdAt: string | null;
}

export interface SubmissionDetail extends ApiSubmission {
  name: string;
  role: string;
  aiTags: string[];
  latitude: number | null;
  longitude: number | null;
  hasImage: boolean;
}

export interface SubmissionAnalysis {
  ok: boolean;
  submissionId: string;
  summary: string;
  recommendation: string[];
  suggestedDepartment: string;
  urgencyRationale: string;
  imageCaption: string;
}

function severityToUrgency(severity: string): Urgency {
  if (severity === "Safety risk") return "safety";
  if (severity === "High priority") return "high";
  return "normal";
}

function issueTypeToCategory(issueType: string): Category {
  const map: Record<string, Category> = {
    Sanitation: "sanitation",
    Drainage: "sanitation",
    Roads: "roads",
    "Water Supply": "water",
    Electricity: "electricity",
  };
  return map[issueType] ?? "other";
}

function urgencyScore(urgency: Urgency): number {
  if (urgency === "safety") return 90;
  if (urgency === "high") return 70;
  return 50;
}

export function submissionToTheme(submission: ApiSubmission): Cluster {
  const urgency = severityToUrgency(submission.severity ?? "");
  const title = submission.topic?.trim() || submission.description?.trim().slice(0, 80) || "Untitled submission";

  return {
    id: submission.id,
    title,
    category: issueTypeToCategory(submission.issueType ?? ""),
    ward: "",
    locality: submission.locality?.trim() || "Location not provided",
    affected: 1,
    urgency,
    status: "new",
    score: urgencyScore(urgency),
    isLiveSubmission: true,
    rationale: {
      demandComponent: 0,
      urgencyComponent: 0,
      dataComponent: 0,
    },
    geo: { x: 50, y: 50 },
    sampleQuotes: [submission.description ?? ""],
    relayShare: submission.submittedFor === "someone_else" ? 1 : 0,
  };
}

export type SubmissionsFetchResult =
  | { ok: true; submissions: ApiSubmission[] }
  | { ok: false; error: string };

export async function fetchSubmissions(): Promise<SubmissionsFetchResult> {
  try {
    const res = await fetchWithTimeout("/api/submissions", { cache: "no-store" });
    const body = await res.text();

    if (!res.ok) {
      console.error(
        `[submissions] GET /api/submissions failed: ${res.status}`,
        body,
      );
      return {
        ok: false,
        error: `Failed to fetch submissions (${res.status})`,
      };
    }

    let data: unknown;
    try {
      data = JSON.parse(body);
    } catch {
      console.error("[submissions] GET /api/submissions returned invalid JSON:", body);
      return { ok: false, error: "Invalid submissions response" };
    }

    return { ok: true, submissions: Array.isArray(data) ? data : [] };
  } catch (err) {
    const message = err instanceof Error ? err.message : "Could not fetch submissions";
    console.error("[submissions] fetch error:", message);
    return { ok: false, error: message };
  }
}

export function isLiveSubmissionCluster(cluster: Cluster): boolean {
  return cluster.isLiveSubmission === true;
}

function listItemToDetail(item: ApiSubmission): SubmissionDetail {
  return {
    ...item,
    name: "",
    role: "",
    aiTags: item.aiTags ?? [],
    latitude: item.latitude ?? null,
    longitude: item.longitude ?? null,
    hasImage: item.hasImage ?? false,
  };
}

async function fetchSubmissionFromList(
  id: string,
): Promise<{ ok: true; submission: SubmissionDetail } | { ok: false; error: string }> {
  const listResult = await fetchSubmissions();
  if (!listResult.ok) {
    return { ok: false, error: listResult.error };
  }

  const match = listResult.submissions.find((item) => item.id === id);
  if (!match) {
    return { ok: false, error: "Issue not found" };
  }

  return { ok: true, submission: listItemToDetail(match) };
}

export async function fetchSubmissionDetail(
  id: string,
): Promise<{ ok: true; submission: SubmissionDetail } | { ok: false; error: string }> {
  try {
    const res = await fetchWithTimeout(`/api/submissions/${encodeURIComponent(id)}`, {
      cache: "no-store",
    });
    if (res.ok) {
      const submission = (await res.json()) as SubmissionDetail;
      return { ok: true, submission };
    }
    if (res.status === 404) {
      return fetchSubmissionFromList(id);
    }
    return {
      ok: false,
      error: `Failed to load issue (${res.status})`,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : "Could not load issue";
    return fetchSubmissionFromList(id).then((fallback) =>
      fallback.ok ? fallback : { ok: false, error: message },
    );
  }
}

export async function fetchSubmissionAnalysis(
  submission: SubmissionDetail,
): Promise<{ ok: true; analysis: SubmissionAnalysis } | { ok: false; error: string }> {
  const id = submission.id;

  try {
    const getRes = await fetchWithTimeout(
      `/api/submissions/${encodeURIComponent(id)}/analysis`,
      { cache: "no-store" },
      90_000,
    );
    if (getRes.ok) {
      const analysis = (await getRes.json()) as SubmissionAnalysis;
      return { ok: true, analysis };
    }
  } catch {
    // Fall through to POST analyze.
  }

  try {
    const postRes = await fetchWithTimeout(
      "/api/submissions/analyze",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: submission.id,
          topic: submission.topic,
          description: submission.description,
          issueType: submission.issueType,
          severity: submission.severity,
          aiTags: submission.aiTags,
          locality: submission.locality,
          submittedFor: submission.submittedFor,
          hasImage: submission.hasImage,
        }),
        cache: "no-store",
      },
      90_000,
    );
    if (!postRes.ok) {
      return {
        ok: false,
        error: postRes.status === 404 ? "Analysis unavailable — run npm run stop, then npm run dev" : `Analysis failed (${postRes.status})`,
      };
    }
    const analysis = (await postRes.json()) as SubmissionAnalysis;
    return { ok: true, analysis };
  } catch (err) {
    const message = err instanceof Error ? err.message : "Could not generate analysis";
    return { ok: false, error: message };
  }
}

export async function fetchSubmissionImage(
  id: string,
): Promise<{ ok: true; imageBase64: string } | { ok: false; error: string }> {
  try {
    const res = await fetchWithTimeout(
      `/api/submissions/${encodeURIComponent(id)}/image`,
      { cache: "no-store" },
    );
    if (!res.ok) {
      return { ok: false, error: "No photo attached" };
    }
    const data = (await res.json()) as { imageBase64?: string | null };
    if (!data.imageBase64) {
      return { ok: false, error: "No photo attached" };
    }
    return { ok: true, imageBase64: data.imageBase64 };
  } catch (err) {
    const message = err instanceof Error ? err.message : "Could not load photo";
    return { ok: false, error: message };
  }
}

export interface ThemeApiResponse extends ApiSubmission {
  affected: number;
  relayShare: number;
  sampleQuotes: string[];
}

export function themeFromApi(theme: ThemeApiResponse): Cluster {
  const cluster = submissionToTheme(theme);
  return {
    ...cluster,
    affected: theme.affected,
    relayShare: theme.relayShare,
    sampleQuotes: theme.sampleQuotes.length ? theme.sampleQuotes : cluster.sampleQuotes,
  };
}

// Backend groups submissions by similarity before returning them (issue
// #30) — multiple citizens reporting the same/similar issue collapse into
// one theme with an incremented voice count, rather than each staying its
// own permanently separate card.
export async function fetchSubmissionThemes(): Promise<{
  themes: Cluster[];
  error: string | null;
}> {
  try {
    const res = await fetchWithTimeout("/api/submissions/themes", { cache: "no-store" });
    const body = await res.text();

    if (!res.ok) {
      console.error(
        `[submissions] GET /api/submissions/themes failed: ${res.status}`,
        body,
      );
      return { themes: [], error: `Failed to fetch submission themes (${res.status})` };
    }

    let data: unknown;
    try {
      data = JSON.parse(body);
    } catch {
      console.error("[submissions] GET /api/submissions/themes returned invalid JSON:", body);
      return { themes: [], error: "Invalid submission themes response" };
    }

    const themes = Array.isArray(data) ? (data as ThemeApiResponse[]) : [];
    return { themes: themes.map(themeFromApi), error: null };
  } catch (err) {
    const message = err instanceof Error ? err.message : "Could not fetch submission themes";
    console.error("[submissions] fetch error:", message);
    return { themes: [], error: message };
  }
}
