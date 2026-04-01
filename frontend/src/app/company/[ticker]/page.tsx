"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
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
      setTimeline(res?.timeline ?? []);
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

  async function handleAnalyze(filingId: number) {
    setAnalyzingId(filingId);
    try {
      await analyzeFile(filingId);
      await fetchAll();
      await fetchTimeline();
    } catch (err) {
      console.error("Analysis failed:", err);
    } finally {
      setAnalyzingId(null);
    }
  }

  const companyName = company?.name ?? overview?.company_name ?? ticker;
  const latestDrift = overview?.latest_drift;
  const totalFilings = overview?.total_filings ?? filings.length;
  const velocity = kinematics?.velocity;
  const alertsCount = overview?.alerts_count ?? 0;

  // Split phrases: newly seen vs recurring
  const sortedPhrases = [...phrases].sort(
    (a, b) =>
      new Date(b.last_seen_date).getTime() -
      new Date(a.last_seen_date).getTime()
  );
  const recentPhrases = sortedPhrases.filter(
    (p) => p.first_seen_date === p.last_seen_date
  );
  const recurringPhrases = sortedPhrases.filter(
    (p) => p.first_seen_date !== p.last_seen_date
  );

  const selectedSectionLabel =
    SECTION_TYPES.find((s) => s.value === timelineSection)?.label ??
    timelineSection;

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] text-[#e5e5e5]">
        <div className="mx-auto max-w-7xl px-6 py-10">
          <div className="animate-pulse space-y-6">
            <div className="h-8 w-64 rounded bg-[#1a1a1a]" />
            <div className="grid grid-cols-4 gap-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="h-24 rounded-lg bg-[#111111] border border-[#1a1a1a]" />
              ))}
            </div>
            <div className="h-72 rounded-lg bg-[#111111] border border-[#1a1a1a]" />
            <div className="h-48 rounded-lg bg-[#111111] border border-[#1a1a1a]" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#e5e5e5]">
      <div className="mx-auto max-w-7xl px-6 py-10 space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold tracking-tight">
              {companyName}
            </h1>
            <span className="rounded bg-[#c8a97e]/10 px-2 py-0.5 text-xs font-mono font-medium text-[#c8a97e]">
              {ticker}
            </span>
          </div>
          <button
            onClick={handleIngest}
            disabled={ingesting}
            className="flex items-center gap-2 rounded-md border border-[#1a1a1a] bg-[#111111] px-3 py-2 text-sm hover:border-[#c8a97e]/40 transition-colors disabled:opacity-50"
          >
            {ingesting ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <FileText className="h-3.5 w-3.5 text-[#737373]" />
            )}
            {ingesting ? "Ingesting..." : "Ingest Filings"}
          </button>
        </div>

        {/* Overview Cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <OverviewCard
            label="Latest Drift"
            value={latestDrift != null ? latestDrift.toFixed(4) : "--"}
            icon={<TrendingUp className="h-4 w-4" />}
            accent={latestDrift != null && latestDrift > 0.1}
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
        <div className="rounded-lg border border-[#1a1a1a] bg-[#111111] p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-sm font-medium text-[#737373] uppercase tracking-wider">
              Drift Timeline
            </h2>
            <div className="relative">
              <button
                onClick={() => setSectionDropdownOpen(!sectionDropdownOpen)}
                className="flex items-center gap-2 rounded-md border border-[#1a1a1a] bg-[#0a0a0a] px-3 py-1.5 text-xs text-[#e5e5e5] hover:border-[#c8a97e]/40 transition-colors"
              >
                {selectedSectionLabel}
                <ChevronDown className="h-3 w-3 text-[#737373]" />
              </button>
              {sectionDropdownOpen && (
                <div className="absolute right-0 top-full mt-1 z-50 min-w-[180px] rounded-md border border-[#1a1a1a] bg-[#111111] py-1">
                  {SECTION_TYPES.map((s) => (
                    <button
                      key={s.value}
                      onClick={() => {
                        setTimelineSection(s.value);
                        setSectionDropdownOpen(false);
                      }}
                      className={`block w-full text-left px-3 py-1.5 text-xs transition-colors ${
                        timelineSection === s.value
                          ? "text-[#c8a97e]"
                          : "text-[#e5e5e5] hover:text-[#c8a97e] hover:bg-[#1a1a1a]"
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
        <div className="rounded-lg border border-[#1a1a1a] bg-[#111111] overflow-hidden">
          <div className="px-6 py-4 border-b border-[#1a1a1a]">
            <h2 className="text-sm font-medium text-[#737373] uppercase tracking-wider">
              Filing History
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#1a1a1a]">
                  <th className="px-6 py-3 text-left text-xs font-medium text-[#737373] uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-[#737373] uppercase tracking-wider">
                    Form
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-[#737373] uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-[#737373] uppercase tracking-wider">
                    Accession
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-[#737373] uppercase tracking-wider">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody>
                {filings.length === 0 ? (
                  <tr>
                    <td
                      colSpan={5}
                      className="px-6 py-12 text-center text-[#737373]"
                    >
                      No filings found. Click &quot;Ingest Filings&quot; to fetch from SEC.
                    </td>
                  </tr>
                ) : (
                  filings.map((f) => (
                    <tr
                      key={f.id}
                      className="border-b border-[#1a1a1a] hover:bg-[#1a1a1a]/50 transition-colors"
                    >
                      <td className="px-6 py-3 text-[#e5e5e5]">
                        {f.filed_date}
                      </td>
                      <td className="px-6 py-3">
                        <span className="rounded bg-[#1a1a1a] px-2 py-0.5 text-xs font-mono">
                          {f.form_type}
                        </span>
                      </td>
                      <td className="px-6 py-3">
                        <StatusBadge processed={f.processed} />
                      </td>
                      <td className="px-6 py-3 font-mono text-xs text-[#737373] truncate max-w-[200px]">
                        {f.accession_number}
                      </td>
                      <td className="px-6 py-3 text-right">
                        <button
                          onClick={() => handleAnalyze(f.id)}
                          disabled={analyzingId === f.id}
                          className="text-xs text-[#c8a97e] hover:text-[#c8a97e]/80 transition-colors disabled:opacity-50"
                        >
                          {analyzingId === f.id ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin inline" />
                          ) : (
                            "Analyze"
                          )}
                        </button>
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
          <div className="rounded-lg border border-[#1a1a1a] bg-[#111111] p-6">
            <div className="flex items-center gap-2 mb-4">
              <ArrowUpRight className="h-4 w-4 text-emerald-400" />
              <h2 className="text-sm font-medium text-[#737373] uppercase tracking-wider">
                New Phrases
              </h2>
            </div>
            <div className="space-y-2">
              {recentPhrases.length === 0 ? (
                <p className="text-xs text-[#737373]">No new phrases detected.</p>
              ) : (
                recentPhrases.slice(0, 20).map((p) => (
                  <div
                    key={p.id}
                    className="flex items-start justify-between gap-3 rounded bg-emerald-400/5 border border-emerald-400/10 px-3 py-2"
                  >
                    <span className="text-xs text-emerald-300">
                      {p.phrase}
                    </span>
                    <span className="shrink-0 text-xs text-[#737373]">
                      {p.first_seen_date}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="rounded-lg border border-[#1a1a1a] bg-[#111111] p-6">
            <div className="flex items-center gap-2 mb-4">
              <ArrowDownRight className="h-4 w-4 text-[#c8a97e]" />
              <h2 className="text-sm font-medium text-[#737373] uppercase tracking-wider">
                Recurring Phrases
              </h2>
            </div>
            <div className="space-y-2">
              {recurringPhrases.length === 0 ? (
                <p className="text-xs text-[#737373]">No recurring phrases found.</p>
              ) : (
                recurringPhrases.slice(0, 20).map((p) => (
                  <div
                    key={p.id}
                    className="flex items-start justify-between gap-3 rounded bg-[#c8a97e]/5 border border-[#c8a97e]/10 px-3 py-2"
                  >
                    <div className="min-w-0">
                      <span className="text-xs text-[#e5e5e5]">{p.phrase}</span>
                      <span className="ml-2 text-[10px] text-[#737373]">x{p.frequency}</span>
                    </div>
                    <span className="shrink-0 text-xs text-[#737373]">
                      {p.first_seen_date} - {p.last_seen_date}
                    </span>
                  </div>
                ))
              )}
            </div>
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
    <div className="rounded-lg border border-[#1a1a1a] bg-[#111111] p-5">
      <div className="flex items-center gap-2 mb-3">
        <span className={accent ? "text-[#c8a97e]" : "text-[#737373]"}>
          {icon}
        </span>
        <span className="text-xs text-[#737373] uppercase tracking-wider">
          {label}
        </span>
      </div>
      <p
        className={`text-lg font-semibold font-mono ${
          accent ? "text-[#c8a97e]" : "text-[#e5e5e5]"
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
      {processed ? "Processed" : "Pending"}
    </span>
  );
}
