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

type SortField = "cosine_distance" | "filing_date" | "name" | "ticker";
type SortDirection = "asc" | "desc";

function SkeletonRow() {
  return (
    <tr className="border-b border-border">
      {Array.from({ length: 5 }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-4 rounded bg-muted animate-pulse" />
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
  const [sortField, setSortField] = useState<SortField>("cosine_distance");
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
    { key: "name", label: "Company", sortable: true, sortKey: "name" },
    { key: "cosine_distance", label: "Drift Score", sortable: true, sortKey: "cosine_distance" },
    { key: "filing_date", label: "Filing Date", sortable: true, sortKey: "filing_date" },
  ];

  const selectedLabel =
    SECTION_TYPES.find((s) => s.value === sectionType)?.label ?? sectionType;

  return (
    <div className="mx-auto max-w-7xl space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-foreground">Screener</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Companies ranked by language drift
          </p>
        </div>

        <div className="relative">
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2 text-sm text-foreground hover:border-accent/40 transition-colors"
          >
            {selectedLabel}
            <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
          </button>
          {dropdownOpen && (
            <div className="absolute right-0 top-full mt-1 z-50 min-w-[180px] rounded-md border border-border bg-card py-1">
              {SECTION_TYPES.map((s) => (
                <button
                  key={s.value}
                  onClick={() => {
                    setSectionType(s.value);
                    setDropdownOpen(false);
                  }}
                  className={`block w-full text-left px-3 py-1.5 text-sm transition-colors ${
                    sectionType === s.value
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

      <div className="rounded-lg border border-border bg-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                {columns.map((col) => (
                  <th
                    key={col.key}
                    className={`px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider ${
                      col.sortable ? "cursor-pointer select-none hover:text-accent transition-colors" : ""
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
                              ? "text-accent"
                              : "text-muted-foreground/50"
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
                      className="border-b border-border cursor-pointer hover:bg-muted/50 transition-colors"
                    >
                      <td className="px-4 py-3 text-muted-foreground font-mono text-xs">
                        {idx + 1}
                      </td>
                      <td className="px-4 py-3 font-mono font-medium text-accent">
                        {row.ticker}
                      </td>
                      <td className="px-4 py-3 text-foreground">{row.name}</td>
                      <td className="px-4 py-3 font-mono text-foreground">
                        {row.cosine_distance.toFixed(4)}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground">
                        {row.filing_date}
                      </td>
                    </tr>
                  ))}
            </tbody>
          </table>
        </div>

        {!loading && sortedData.length === 0 && (
          <div className="flex items-center justify-center py-16 text-sm text-muted-foreground">
            No data available for this section type.
          </div>
        )}
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
