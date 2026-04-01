const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types — loosely typed to match backend JSON responses
// ---------------------------------------------------------------------------

export interface Company {
  cik: string;
  ticker: string;
  name: string;
  in_database?: boolean;
  id?: number;
}

export interface Filing {
  id?: number;
  accession_number: string;
  form_type: string;
  filing_date: string;
  report_date?: string;
  status?: string;
  sections?: { type: string; word_count: number }[];
  document_url?: string;
  db_status?: string | null;
  primary_document?: string;
}

export interface Section {
  filing_id: number;
  section_type: string;
  word_count: number;
  text: string;
}

export interface DriftScore {
  section_type: string;
  cosine_distance: number;
  jaccard_distance: number;
  added_words: number;
  removed_words: number;
  sentiment_delta: Record<string, number>;
  filing_date: string;
  form_type: string;
  accession_number: string;
}

export interface SentenceChange {
  change_type: string;
  text: string;
  matched_text?: string;
  similarity?: number;
  index?: number;
}

export interface ScreenerEntry {
  ticker: string;
  name: string;
  cosine_distance: number;
  jaccard_distance: number;
  added_words: number;
  removed_words: number;
  sentiment_delta: Record<string, number>;
  filing_date: string;
}

export interface Phrase {
  phrase: string;
  section_type: string;
  status: string;
  filing_date: string;
  form_type: string;
}

export interface Alert {
  id: number;
  alert_type: string;
  severity: string;
  message: string;
  metadata?: Record<string, unknown>;
  read: boolean;
  created_at: string;
  ticker: string;
  company_name: string;
  filing_date: string;
}

export interface Watchlist {
  id: number;
  name: string;
  created_at: string;
}

export interface ObfuscationResult {
  section_type: string;
  overall_obfuscation_score: number;
  density_change: number;
  specificity_change: number;
  readability_change: number;
  detected_euphemisms: string[];
}

export interface EntropyResult {
  section_type: string;
  kl_divergence: number;
  novelty_score: number;
  entropy_rate_change: number;
  vocab_overlap: number;
}

export interface KinematicsResult {
  [section: string]: {
    phase: string;
    latest_velocity: number;
    latest_acceleration: number;
    latest_momentum: number;
    velocity_mean: number;
    periods_analyzed: number;
    note?: string;
  };
}

export interface OverviewResult {
  ticker: string;
  company: string;
  latest_drift: Record<string, Record<string, unknown>>;
  anomaly: Record<string, Record<string, unknown>>;
  kinematics: Record<string, Record<string, unknown>>;
  trends: Record<string, unknown>;
}

export interface IngestResult {
  ticker: string;
  ingested: { accession: string; form_type: string; filing_date: string; sections: string[]; status: string }[];
  count: number;
}

export interface AnalyzeResult {
  filing_id: number;
  prev_filing_id: number;
  form_type: string;
  sections_analyzed: number;
  results: Record<string, unknown>[];
  anomaly?: Record<string, unknown>;
  trends?: Record<string, unknown>;
  alerts_created?: Record<string, unknown>[];
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  status: number;
  detail?: string;

  constructor(status: number, message: string, detail?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = await res.json();
      detail = body.detail ?? body.message ?? JSON.stringify(body);
    } catch {
      detail = res.statusText;
    }
    throw new ApiError(res.status, `API error ${res.status}: ${detail}`, detail);
  }

  if (res.status === 204) {
    return undefined as T;
  }

  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | boolean | undefined>): string {
  const filtered = Object.entries(params).filter(
    ([, v]) => v !== undefined && v !== null
  );
  if (filtered.length === 0) return "";
  return "?" + new URLSearchParams(
    filtered.map(([k, v]) => [k, String(v)])
  ).toString();
}

// ---------------------------------------------------------------------------
// Companies — GET /companies, GET /companies/{ticker}
// ---------------------------------------------------------------------------

export async function searchCompanies(q: string) {
  const res = await request<{ data: Company[]; total: number }>(`/companies${qs({ q })}`);
  return res.data;
}

export async function getCompany(ticker: string) {
  return request<Company>(`/companies/${encodeURIComponent(ticker)}`);
}

// ---------------------------------------------------------------------------
// Filings — GET /companies/{ticker}/filings, POST /filings/ingest/{ticker}
// ---------------------------------------------------------------------------

export async function getCompanyFilings(ticker: string, formType?: string) {
  const res = await request<{ data: Filing[]; total: number }>(
    `/companies/${encodeURIComponent(ticker)}/filings${qs({ form_type: formType })}`
  );
  return res.data;
}

export async function ingestFilings(ticker: string, formType: string, limit: number) {
  return request<IngestResult>(
    `/filings/ingest/${encodeURIComponent(ticker)}${qs({ form_type: formType, limit })}`,
    { method: "POST" }
  );
}

export async function getFiling(id: number) {
  return request<Filing>(`/filings/${id}`);
}

export async function getSection(filingId: number, sectionType: string) {
  return request<Section>(
    `/filings/${filingId}/sections/${encodeURIComponent(sectionType)}`
  );
}

// ---------------------------------------------------------------------------
// Analysis — POST /filings/{id}/analyze
// ---------------------------------------------------------------------------

export async function analyzeFile(filingId: number, force?: boolean) {
  return request<AnalyzeResult>(
    `/filings/${filingId}/analyze${qs({ force })}`,
    { method: "POST" }
  );
}

// ---------------------------------------------------------------------------
// Drift — GET /companies/{ticker}/drift, GET /filings/{id}/diff
// ---------------------------------------------------------------------------

export async function getDriftTimeline(ticker: string, sectionType?: string) {
  const res = await request<{ ticker: string; data: DriftScore[]; total: number }>(
    `/companies/${encodeURIComponent(ticker)}/drift${qs({ section_type: sectionType })}`
  );
  return res.data;
}

export async function getFilingDiff(filingId: number, vsId: number, sectionType: string) {
  return request<{ filing_id: number; vs_filing_id: number; section_type: string; diff: string; stats: Record<string, unknown> }>(
    `/filings/${filingId}/diff${qs({ vs: vsId, section_type: sectionType })}`
  );
}

export async function getSentenceChanges(driftScoreId: number, changeType?: string) {
  const res = await request<{ drift_score_id: number; data: SentenceChange[]; total: number }>(
    `/drift/${driftScoreId}/sentences${qs({ change_type: changeType })}`
  );
  return res.data;
}

// ---------------------------------------------------------------------------
// Screener — GET /drift/screener
// ---------------------------------------------------------------------------

export async function getScreener(sectionType: string, sortBy?: string, limit?: number) {
  const res = await request<{ data: ScreenerEntry[]; total: number }>(
    `/drift/screener${qs({ section_type: sectionType, sort_by: sortBy, limit })}`
  );
  return res.data;
}

// ---------------------------------------------------------------------------
// Phrases — GET /companies/{ticker}/phrases
// ---------------------------------------------------------------------------

export async function getPhrases(ticker: string) {
  const res = await request<{ ticker: string; data: Phrase[]; total: number }>(
    `/companies/${encodeURIComponent(ticker)}/phrases`
  );
  return res.data;
}

// ---------------------------------------------------------------------------
// Alerts — GET /alerts, PUT /alerts/{id}/read
// ---------------------------------------------------------------------------

export async function getAlerts(unread?: boolean) {
  const res = await request<{ data: Alert[]; total: number }>(
    `/alerts${qs({ unread })}`
  );
  return res.data;
}

export async function markAlertRead(alertId: number) {
  return request<{ id: number; read: boolean }>(
    `/alerts/${alertId}/read`,
    { method: "PUT" }
  );
}

// ---------------------------------------------------------------------------
// Watchlists — POST /watchlists, GET /watchlists, POST /watchlists/{id}/companies
// ---------------------------------------------------------------------------

export async function createWatchlist(name: string) {
  return request<Watchlist>(`/watchlists`, {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export async function getWatchlists() {
  const res = await request<{ data: Watchlist[]; total: number }>(`/watchlists`);
  return res.data;
}

export async function addToWatchlist(watchlistId: number, ticker: string) {
  return request<{ watchlist_id: number; ticker: string; company_id: number }>(
    `/watchlists/${watchlistId}/companies`,
    { method: "POST", body: JSON.stringify({ ticker }) }
  );
}

// ---------------------------------------------------------------------------
// Research — POST /research/obfuscation/{id}, POST /research/entropy/{id},
//            GET /research/kinematics/{ticker}, GET /research/overview/{ticker}
// ---------------------------------------------------------------------------

export async function getObfuscation(filingId: number) {
  return request<{ filing_id: number; sections: ObfuscationResult[] }>(
    `/research/obfuscation/${filingId}`,
    { method: "POST" }
  );
}

export async function getEntropy(filingId: number) {
  return request<{ filing_id: number; sections: EntropyResult[] }>(
    `/research/entropy/${filingId}`,
    { method: "POST" }
  );
}

export async function getKinematics(ticker: string) {
  const res = await request<{ ticker: string; data: KinematicsResult }>(
    `/research/kinematics/${encodeURIComponent(ticker)}`
  );
  return res.data;
}

export async function getOverview(ticker: string) {
  return request<OverviewResult>(
    `/research/overview/${encodeURIComponent(ticker)}`
  );
}
