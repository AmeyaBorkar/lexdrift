"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { getScreener } from "@/lib/api";
import type { ScreenerEntry } from "@/lib/api";
import { ArrowUpDown, ChevronDown } from "lucide-react";

const SECTION_TYPES = [
  { value: "risk_factors", label: "Risk Factors" },
  { value: "mdna", label: "MD&A" },
  { value: "business", label: "Business" },
  { value: "legal_proceedings", label: "Legal Proceedings" },
  { value: "quantitative_disclosures", label: "Quantitative Disclosures" },
];

type SortField = "latest_drift_score" | "filed_date" | "company_name" | "ticker";
type SortDirection = "asc" | "desc";

function SkeletonRow() {
  return (
    <tr className="border-b border-[#1a1a1a]">
      {Array.from({ length: 5 }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-4 rounded bg-[#1a1a1a] animate-pulse" />
        </td>
      ))}
    </tr>
  );
}

export default function ScreenerPage() {
  const router = useRouter();
  const [sectionType, setSectionType] = useState("risk_factors");
  const [data, setData] = useState<ScreenerEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortField, setSortField] = useState<SortField>("latest_drift_score");
  const [sortDir, setSortDir] = useState<SortDirection>("desc");
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getScreener(sectionType, sortField);
      setData(result ?? []);
    } catch (err) {
      console.error("Failed to fetch screener data:", err);
      setData([]);
    } finally {
      setLoading(false);
    }
  }, [sectionType, sortField]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const sortedData = [...data].sort((a, b) => {
    const aVal = a[sortField as keyof ScreenerEntry];
    const bVal = b[sortField as keyof ScreenerEntry];
    if (typeof aVal === "number" && typeof bVal === "number") {
      return sortDir === "desc" ? bVal - aVal : aVal - bVal;
    }
    const aStr = String(aVal ?? "");
    const bStr = String(bVal ?? "");
    return sortDir === "desc"
      ? bStr.localeCompare(aStr)
      : aStr.localeCompare(bStr);
  });

  function handleSort(field: SortField) {
    if (sortField === field) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  }

  const columns: { key: string; label: string; sortable: boolean; sortKey?: SortField }[] = [
    { key: "rank", label: "#", sortable: false },
    { key: "ticker", label: "Ticker", sortable: true, sortKey: "ticker" },
    { key: "company_name", label: "Company", sortable: true, sortKey: "company_name" },
    { key: "latest_drift_score", label: "Drift Score", sortable: true, sortKey: "latest_drift_score" },
    { key: "filed_date", label: "Filing Date", sortable: true, sortKey: "filed_date" },
  ];

  const selectedLabel =
    SECTION_TYPES.find((s) => s.value === sectionType)?.label ?? sectionType;

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#e5e5e5]">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Screener</h1>
            <p className="text-sm text-[#737373] mt-1">
              Companies ranked by language drift
            </p>
          </div>

          <div className="relative">
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center gap-2 rounded-md border border-[#1a1a1a] bg-[#111111] px-3 py-2 text-sm text-[#e5e5e5] hover:border-[#c8a97e]/40 transition-colors"
            >
              {selectedLabel}
              <ChevronDown className="h-3.5 w-3.5 text-[#737373]" />
            </button>
            {dropdownOpen && (
              <div className="absolute right-0 top-full mt-1 z-50 min-w-[180px] rounded-md border border-[#1a1a1a] bg-[#111111] py-1">
                {SECTION_TYPES.map((s) => (
                  <button
                    key={s.value}
                    onClick={() => {
                      setSectionType(s.value);
                      setDropdownOpen(false);
                    }}
                    className={`block w-full text-left px-3 py-1.5 text-sm transition-colors ${
                      sectionType === s.value
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

        <div className="rounded-lg border border-[#1a1a1a] bg-[#111111] overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#1a1a1a]">
                  {columns.map((col) => (
                    <th
                      key={col.key}
                      className={`px-4 py-3 text-left text-xs font-medium text-[#737373] uppercase tracking-wider ${
                        col.sortable ? "cursor-pointer select-none hover:text-[#c8a97e] transition-colors" : ""
                      }`}
                      onClick={() =>
                        col.sortable && col.sortKey && handleSort(col.sortKey)
                      }
                    >
                      <span className="flex items-center gap-1">
                        {col.label}
                        {col.sortable && col.sortKey && (
                          <ArrowUpDown
                            className={`h-3 w-3 ${
                              sortField === col.sortKey
                                ? "text-[#c8a97e]"
                                : "text-[#737373]/50"
                            }`}
                          />
                        )}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {loading
                  ? Array.from({ length: 12 }).map((_, i) => (
                      <SkeletonRow key={i} />
                    ))
                  : sortedData.map((row, idx) => (
                      <tr
                        key={row.ticker + idx}
                        onClick={() => router.push(`/company/${row.ticker}`)}
                        className="border-b border-[#1a1a1a] cursor-pointer hover:bg-[#1a1a1a]/50 transition-colors"
                      >
                        <td className="px-4 py-3 text-[#737373] font-mono text-xs">
                          {idx + 1}
                        </td>
                        <td className="px-4 py-3 font-mono font-medium text-[#c8a97e]">
                          {row.ticker}
                        </td>
                        <td className="px-4 py-3">{row.company_name}</td>
                        <td className="px-4 py-3 font-mono">
                          {row.latest_drift_score.toFixed(4)}
                        </td>
                        <td className="px-4 py-3 text-[#737373]">
                          {row.filed_date}
                        </td>
                      </tr>
                    ))}
              </tbody>
            </table>
          </div>

          {!loading && sortedData.length === 0 && (
            <div className="flex items-center justify-center py-16 text-sm text-[#737373]">
              No data available for this section type.
            </div>
          )}
        </div>
      </div>

      {dropdownOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setDropdownOpen(false)}
        />
      )}
    </div>
  );
}
