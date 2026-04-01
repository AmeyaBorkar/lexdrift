const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Company {
  ticker: string;
  name: string;
  cik: string;
  sector?: string;
  sic_code?: string;
}

export interface Filing {
  id: number;
  company_ticker: string;
  accession_number: string;
  form_type: string;
  filed_date: string;
  period_of_report?: string;
  filing_url?: string;
  processed: boolean;
}

export interface Section {
  id: number;
  filing_id: number;
  section_type: string;
  content: string;
  word_count?: number;
}

export interface DriftScore {
  id: number;
  filing_id: number;
  previous_filing_id: number;
  section_type: string;
  similarity_score: number;
  drift_score: number;
  filed_date: string;
  period_of_report?: string;
}

export interface DriftTimeline {
  ticker: string;
  section_type: string;
  timeline: DriftScore[];
}

export interface FilingDiff {
  filing_id: number;
  vs_filing_id: number;
  section_type: string;
  additions: string[];
  deletions: string[];
  modifications: DiffModification[];
}

export interface DiffModification {
  old_text: string;
  new_text: string;
  similarity: number;
}

export interface SentenceChange {
  id: number;
  drift_score_id: number;
  change_type: string;
  old_text?: string;
  new_text?: string;
  similarity_score?: number;
}

export interface ScreenerEntry {
  ticker: string;
  company_name: string;
  latest_drift_score: number;
  section_type: string;
  filed_date: string;
  trend?: string;
}

export interface Phrase {
  id: number;
  ticker: string;
  phrase: string;
  first_seen_date: string;
  last_seen_date: string;
  frequency: number;
  section_type?: string;
}

export interface Alert {
  id: number;
  ticker: string;
  alert_type: string;
  severity: string;
  message: string;
  created_at: string;
  read: boolean;
  drift_score_id?: number;
}

export interface Watchlist {
  id: number;
  name: string;
  created_at: string;
  tickers: string[];
}

export interface ObfuscationResult {
  filing_id: number;
  score: number;
  details: Record<string, unknown>;
}

export interface EntropyResult {
  filing_id: number;
  entropy: number;
  section_entropies: Record<string, number>;
}

export interface KinematicsResult {
  ticker: string;
  velocity: number;
  acceleration: number;
  timeline: { date: string; velocity: number; acceleration: number }[];
}

export interface OverviewResult {
  ticker: string;
  company_name: string;
  total_filings: number;
  latest_drift?: number;
  avg_drift?: number;
  alerts_count: number;
  latest_filing_date?: string;
}

export interface IngestResult {
  ticker: string;
  form_type: string;
  filings_ingested: number;
}

export interface AnalyzeResult {
  filing_id: number;
  status: string;
  sections_analyzed?: number;
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

  // Handle 204 No Content
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
// Company endpoints
// ---------------------------------------------------------------------------

export async function searchCompanies(q: string): Promise<Company[]> {
  return request<Company[]>(`/api/companies/search${qs({ q })}`);
}

export async function getCompany(ticker: string): Promise<Company> {
  return request<Company>(`/api/companies/${encodeURIComponent(ticker)}`);
}

// ---------------------------------------------------------------------------
// Filing endpoints
// ---------------------------------------------------------------------------

export async function getCompanyFilings(
  ticker: string,
  formType?: string
): Promise<Filing[]> {
  return request<Filing[]>(
    `/api/companies/${encodeURIComponent(ticker)}/filings${qs({ form_type: formType })}`
  );
}

export async function ingestFilings(
  ticker: string,
  formType: string,
  limit: number
): Promise<IngestResult> {
  return request<IngestResult>(`/api/filings/ingest`, {
    method: "POST",
    body: JSON.stringify({ ticker, form_type: formType, limit }),
  });
}

export async function getFiling(id: number): Promise<Filing> {
  return request<Filing>(`/api/filings/${id}`);
}

export async function getSection(
  filingId: number,
  sectionType: string
): Promise<Section> {
  return request<Section>(
    `/api/filings/${filingId}/sections/${encodeURIComponent(sectionType)}`
  );
}

export async function analyzeFile(
  filingId: number,
  force?: boolean
): Promise<AnalyzeResult> {
  return request<AnalyzeResult>(
    `/api/filings/${filingId}/analyze${qs({ force })}`,
    { method: "POST" }
  );
}

// ---------------------------------------------------------------------------
// Drift / diff endpoints
// ---------------------------------------------------------------------------

export async function getDriftTimeline(
  ticker: string,
  sectionType?: string
): Promise<DriftTimeline> {
  return request<DriftTimeline>(
    `/api/drift/${encodeURIComponent(ticker)}/timeline${qs({ section_type: sectionType })}`
  );
}

export async function getFilingDiff(
  filingId: number,
  vsId: number,
  sectionType: string
): Promise<FilingDiff> {
  return request<FilingDiff>(
    `/api/drift/diff/${filingId}/${vsId}${qs({ section_type: sectionType })}`
  );
}

export async function getSentenceChanges(
  driftScoreId: number,
  changeType?: string
): Promise<SentenceChange[]> {
  return request<SentenceChange[]>(
    `/api/drift/changes/${driftScoreId}${qs({ change_type: changeType })}`
  );
}

// ---------------------------------------------------------------------------
// Screener
// ---------------------------------------------------------------------------

export async function getScreener(
  sectionType: string,
  sortBy?: string,
  limit?: number
): Promise<ScreenerEntry[]> {
  return request<ScreenerEntry[]>(
    `/api/screener${qs({ section_type: sectionType, sort_by: sortBy, limit })}`
  );
}

// ---------------------------------------------------------------------------
// Phrases
// ---------------------------------------------------------------------------

export async function getPhrases(ticker: string): Promise<Phrase[]> {
  return request<Phrase[]>(
    `/api/phrases/${encodeURIComponent(ticker)}`
  );
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------

export async function getAlerts(unread?: boolean): Promise<Alert[]> {
  return request<Alert[]>(`/api/alerts${qs({ unread })}`);
}

export async function markAlertRead(alertId: number): Promise<Alert> {
  return request<Alert>(`/api/alerts/${alertId}/read`, { method: "POST" });
}

// ---------------------------------------------------------------------------
// Watchlists
// ---------------------------------------------------------------------------

export async function createWatchlist(name: string): Promise<Watchlist> {
  return request<Watchlist>(`/api/watchlists`, {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export async function getWatchlists(): Promise<Watchlist[]> {
  return request<Watchlist[]>(`/api/watchlists`);
}

export async function addToWatchlist(
  watchlistId: number,
  ticker: string
): Promise<Watchlist> {
  return request<Watchlist>(`/api/watchlists/${watchlistId}/add`, {
    method: "POST",
    body: JSON.stringify({ ticker }),
  });
}

// ---------------------------------------------------------------------------
// Advanced analysis
// ---------------------------------------------------------------------------

export async function getObfuscation(filingId: number): Promise<ObfuscationResult> {
  return request<ObfuscationResult>(`/api/analysis/obfuscation/${filingId}`);
}

export async function getEntropy(filingId: number): Promise<EntropyResult> {
  return request<EntropyResult>(`/api/analysis/entropy/${filingId}`);
}

export async function getKinematics(ticker: string): Promise<KinematicsResult> {
  return request<KinematicsResult>(
    `/api/analysis/kinematics/${encodeURIComponent(ticker)}`
  );
}

export async function getOverview(ticker: string): Promise<OverviewResult> {
  return request<OverviewResult>(
    `/api/companies/${encodeURIComponent(ticker)}/overview`
  );
}
