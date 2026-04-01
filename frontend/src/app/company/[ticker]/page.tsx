"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  getCompany,
  getCompanyFilings,
  getDriftTimeline,
  getPhrases,
  getOverview,
  getKinematics,
  ingestFilings,
  analyzeFile,
  type Company,
  type Filing,
  type Phrase,
  type DriftScore,
  type OverviewResult,
  type KinematicsResult,
} from "@/lib/api";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowLeft,
  ArrowUpRight,
  ChevronDown,
  FileText,
  Loader2,
  TrendingUp,
} from "lucide-react";
import DriftChart from "./drift-chart";

const SECTION_TYPES = [
  { value: "risk_factors", label: "Risk Factors" },
  { value: "mdna", label: "MD&A" },
  { value: "business", label: "Business" },
  { value: "legal_proceedings", label: "Legal Proceedings" },
  { value: "quantitative_disclosures", label: "Quantitative Disclosures" },
];

export default function CompanyPage() {
  const params = useParams();
  const ticker = (params.ticker as string)?.toUpperCase() ?? "";

  const [company, setCompany] = useState<Company | null>(null);
  const [overview, setOverview] = useState<OverviewResult | null>(null);
  const [kinematics, setKinematics] = useState<KinematicsResult | null>(null);
  const [filings, setFilings] = useState<Filing[]>([]);
  const [phrases, setPhrases] = useState<Phrase[]>([]);
  const [timeline, setTimeline] = useState<DriftScore[]>([]);
  const [timelineSection, setTimelineSection] = useState("risk_factors");
  const [sectionDropdownOpen, setSectionDropdownOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [ingesting, setIngesting] = useState(false);
  const [analyzingId, setAnalyzingId] = useState<number | null>(null);

  const fetchAll = useCallback(async () => {
    if (!ticker) return;
    setLoading(true);
    try {
      const [companyRes, overviewRes, kinematicsRes, filingsRes, phrasesRes] =
        await Promise.allSettled([
          getCompany(ticker),
          getOverview(ticker),
          getKinematics(ticker),
          getCompanyFilings(ticker),
          getPhrases(ticker),
        ]);

      if (companyRes.status === "fulfilled") setCompany(companyRes.value);
      if (overviewRes.status === "fulfilled") setOverview(overviewRes.value);
      if (kinematicsRes.status === "fulfilled") setKinematics(kinematicsRes.value);
      if (filingsRes.status === "fulfilled") setFilings(filingsRes.value ?? []);
      if (phrasesRes.status === "fulfilled") setPhrases(phrasesRes.value ?? []);
    } catch (err) {
      console.error("Failed to load company data:", err);
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  const fetchTimeline = useCallback(async () => {
    if (!ticker) return;
    try {
      const res = await getDriftTimeline(ticker, timelineSection);
      setTimeline(res ?? []);
    } catch {
      setTimeline([]);
    }
  }, [ticker, timelineSection]);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  useEffect(() => {
    fetchTimeline();
  }, [fetchTimeline]);

  async function handleIngest() {
    setIngesting(true);
    try {
      await ingestFilings(ticker, "10-K", 5);
      await fetchAll();
    } catch (err) {
      console.error("Ingest failed:", err);
    } finally {
      setIngesting(false);
    }
  }

  async function handleAnalyze(filingId: number, alreadyAnalyzed: boolean = false) {
    setAnalyzingId(filingId);
    try {
      await analyzeFile(filingId, alreadyAnalyzed ? true : undefined);
      await fetchAll();
      await fetchTimeline();
    } catch (err) {
      console.error("Analysis failed:", err);
    } finally {
      setAnalyzingId(null);
    }
  }

  const companyName = company?.name ?? overview?.company ?? ticker;
  const latestDrift = overview?.latest_drift;
  const totalFilings = overview?.latest_drift ? Object.keys(overview.latest_drift).length : filings.length;
  const firstSection = kinematics ? Object.values(kinematics)[0] : undefined;
  const velocity = firstSection?.latest_velocity;
  const alertsCount = 0;

  // Group phrases by filing date, split new vs recurring
  const sortedPhrases = [...phrases].sort(
    (a, b) =>
      new Date(b.filing_date).getTime() -
      new Date(a.filing_date).getTime()
  );
  const recentPhrases = sortedPhrases.filter(
    (p) => p.status === "added"
  );
  const recurringPhrases = sortedPhrases.filter(
    (p) => p.status !== "added"
  );

  // Group new phrases by filing date
  const phrasesByDate = recentPhrases.reduce<Record<string, Phrase[]>>((acc, p) => {
    (acc[p.filing_date] ??= []).push(p);
    return acc;
  }, {});

  const selectedSectionLabel =
    SECTION_TYPES.find((s) => s.value === timelineSection)?.label ??
    timelineSection;

  if (loading) {
    return (
      <div className="mx-auto max-w-7xl space-y-6">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to Dashboard
        </Link>
        <div className="animate-pulse space-y-6">
          <div className="h-8 w-64 rounded bg-muted" />
          <div className="grid grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-24 rounded-lg bg-card border border-border" />
            ))}
          </div>
          <div className="h-72 rounded-lg bg-card border border-border" />
          <div className="h-48 rounded-lg bg-card border border-border" />
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl space-y-8">
      {/* Back link */}
      <Link
        href="/"
        className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Back to Dashboard
      </Link>

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold tracking-tight text-foreground">
            {companyName}
          </h1>
          <span className="rounded bg-accent/10 px-2 py-0.5 text-xs font-mono font-medium text-accent">
            {ticker}
          </span>
        </div>
        <button
          onClick={handleIngest}
          disabled={ingesting}
          className="flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2 text-sm text-foreground hover:border-accent/40 transition-colors disabled:opacity-50"
        >
          {ingesting ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <FileText className="h-3.5 w-3.5 text-muted-foreground" />
          )}
          {ingesting ? "Ingesting..." : "Ingest Filings"}
        </button>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <OverviewCard
          label="Latest Drift"
          value={latestDrift != null ? String(Object.keys(latestDrift).length) + " sections" : "--"}
          icon={<TrendingUp className="h-4 w-4" />}
          accent={latestDrift != null && Object.keys(latestDrift).length > 0}
        />
        <OverviewCard
          label="Active Alerts"
          value={String(alertsCount)}
          icon={<AlertTriangle className="h-4 w-4" />}
          accent={alertsCount > 0}
        />
        <OverviewCard
          label="Drift Velocity"
          value={velocity != null ? velocity.toFixed(4) : "--"}
          icon={<Activity className="h-4 w-4" />}
        />
        <OverviewCard
          label="Total Filings"
          value={String(totalFilings)}
          icon={<FileText className="h-4 w-4" />}
        />
      </div>

      {/* Drift Timeline Chart */}
      <div className="rounded-lg border border-border bg-card p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            Drift Timeline
          </h2>
          <div className="relative">
            <button
              onClick={() => setSectionDropdownOpen(!sectionDropdownOpen)}
              className="flex items-center gap-2 rounded-md border border-border bg-background px-3 py-1.5 text-xs text-foreground hover:border-accent/40 transition-colors"
            >
              {selectedSectionLabel}
              <ChevronDown className="h-3 w-3 text-muted-foreground" />
            </button>
            {sectionDropdownOpen && (
              <div className="absolute right-0 top-full mt-1 z-50 min-w-[180px] rounded-md border border-border bg-card py-1">
                {SECTION_TYPES.map((s) => (
                  <button
                    key={s.value}
                    onClick={() => {
                      setTimelineSection(s.value);
                      setSectionDropdownOpen(false);
                    }}
                    className={`block w-full text-left px-3 py-1.5 text-xs transition-colors ${
                      timelineSection === s.value
                        ? "text-accent"
                        : "text-foreground hover:text-accent hover:bg-muted"
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        <DriftChart data={timeline} />
      </div>

      {/* Filing History */}
      <div className="rounded-lg border border-border bg-card overflow-hidden">
        <div className="px-6 py-4 border-b border-border">
          <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            Filing History
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Date
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Form
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Accession
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Action
                </th>
              </tr>
            </thead>
            <tbody>
              {filings.length === 0 ? (
                <tr>
                  <td
                    colSpan={5}
                    className="px-6 py-12 text-center text-muted-foreground"
                  >
                    No filings found. Click &quot;Ingest Filings&quot; to fetch from SEC.
                  </td>
                </tr>
              ) : (
                filings.map((f) => (
                  <tr
                    key={f.accession_number}
                    className="border-b border-border hover:bg-muted/50 transition-colors"
                  >
                    <td className="px-6 py-3 text-foreground">
                      {f.filing_date}
                    </td>
                    <td className="px-6 py-3">
                      <span className="rounded bg-muted px-2 py-0.5 text-xs font-mono text-foreground">
                        {f.form_type}
                      </span>
                    </td>
                    <td className="px-6 py-3">
                      <StatusBadge processed={f.db_status === "analyzed"} />
                    </td>
                    <td className="px-6 py-3 font-mono text-xs text-muted-foreground truncate max-w-[200px]">
                      {f.accession_number}
                    </td>
                    <td className="px-6 py-3 text-right">
                      {f.id ? (
                        <button
                          onClick={() => handleAnalyze(f.id!, f.db_status === "analyzed")}
                          disabled={analyzingId === f.id}
                          className="text-xs text-accent hover:text-accent/80 transition-colors disabled:opacity-50"
                        >
                          {analyzingId === f.id ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin inline" />
                          ) : f.db_status === "analyzed" ? (
                            "Re-analyze"
                          ) : (
                            "Analyze"
                          )}
                        </button>
                      ) : (
                        <span className="text-xs text-muted-foreground">Not ingested</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Key Phrases */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-lg border border-border bg-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <ArrowUpRight className="h-4 w-4 text-emerald-400" />
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              New Phrases
            </h2>
          </div>
          <div className="space-y-4">
            {Object.keys(phrasesByDate).length === 0 ? (
              <p className="text-xs text-muted-foreground">No new phrases detected.</p>
            ) : (
              Object.entries(phrasesByDate).slice(0, 5).map(([date, datePhrases]) => (
                <div key={date}>
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">
                    {date}
                  </p>
                  <div className="space-y-1.5">
                    {datePhrases.slice(0, 8).map((p, idx) => (
                      <div
                        key={`${p.phrase}-${idx}`}
                        className="rounded bg-emerald-400/5 border border-emerald-400/10 px-3 py-1.5"
                      >
                        <span className="text-xs text-emerald-300 dark:text-emerald-300 text-emerald-700">
                          {p.phrase}
                        </span>
                      </div>
                    ))}
                    {datePhrases.length > 8 && (
                      <p className="text-[10px] text-muted-foreground">
                        +{datePhrases.length - 8} more
                      </p>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <ArrowDownRight className="h-4 w-4 text-accent" />
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              Recurring Phrases
            </h2>
          </div>
          <div className="space-y-2">
            {recurringPhrases.length === 0 ? (
              <p className="text-xs text-muted-foreground">No recurring phrases found.</p>
            ) : (
              recurringPhrases.slice(0, 20).map((p, idx) => (
                <div
                  key={`${p.phrase}-${idx}`}
                  className="flex items-start justify-between gap-3 rounded bg-accent/5 border border-accent/10 px-3 py-2"
                >
                  <div className="min-w-0">
                    <span className="text-xs text-foreground">{p.phrase}</span>
                    <span className="ml-2 text-[10px] text-muted-foreground">{p.status}</span>
                  </div>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {p.filing_date}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {sectionDropdownOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setSectionDropdownOpen(false)}
        />
      )}
    </div>
  );
}

function OverviewCard({
  label,
  value,
  icon,
  accent,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="flex items-center gap-2 mb-3">
        <span className={accent ? "text-accent" : "text-muted-foreground"}>
          {icon}
        </span>
        <span className="text-xs text-muted-foreground uppercase tracking-wider">
          {label}
        </span>
      </div>
      <p
        className={`text-lg font-semibold font-mono ${
          accent ? "text-accent" : "text-foreground"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function StatusBadge({ processed }: { processed: boolean }) {
  return (
    <span
      className={`rounded px-2 py-0.5 text-xs font-medium ${
        processed
          ? "text-emerald-400 bg-emerald-400/10"
          : "text-yellow-400 bg-yellow-400/10"
      }`}
    >
      {processed ? "Analyzed" : "Pending"}
    </span>
  );
}
