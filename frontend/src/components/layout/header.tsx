"use client";

import { Bell, Search } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState, useEffect, useRef, useCallback } from "react";
import { searchCompanies, getAlerts, type Company } from "@/lib/api";
import { cn } from "@/lib/utils";

export function Header() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Company[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Fetch unread alerts count
  useEffect(() => {
    let cancelled = false;
    async function fetchAlerts() {
      try {
        const alerts = await getAlerts(true);
        if (!cancelled) setUnreadCount(alerts.length);
      } catch {
        // Silently fail — API may not be running
      }
    }
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  // Search with debounce
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handleSearch = useCallback((value: string) => {
    setQuery(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!value.trim()) {
      setResults([]);
      setShowResults(false);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      try {
        const data = await searchCompanies(value.trim());
        setResults(data);
        setShowResults(true);
      } catch {
        setResults([]);
      }
    }, 250);
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setShowResults(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  function selectCompany(ticker: string) {
    setShowResults(false);
    setQuery("");
    router.push(`/company/${ticker}`);
  }

  return (
    <header className="sticky top-0 z-20 flex h-14 items-center justify-between border-b border-border bg-card/80 backdrop-blur-sm px-6">
      {/* Spacer for mobile menu button */}
      <div className="w-10 lg:hidden" />

      {/* Search */}
      <div ref={wrapperRef} className="relative mx-auto w-full max-w-md">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={query}
            onChange={(e) => handleSearch(e.target.value)}
            onFocus={() => results.length > 0 && setShowResults(true)}
            placeholder="Search companies..."
            className={cn(
              "h-9 w-full rounded border border-border bg-background pl-9 pr-3 text-sm text-foreground",
              "placeholder:text-muted-foreground",
              "outline-none focus:border-accent/50 transition-colors duration-200"
            )}
          />
        </div>

        {/* Results dropdown */}
        {showResults && results.length > 0 && (
          <div className="absolute top-full left-0 mt-1 w-full rounded border border-border bg-card shadow-sm z-50">
            {results.map((company) => (
              <button
                key={company.ticker}
                onClick={() => selectCompany(company.ticker)}
                className="flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm transition-colors duration-200 hover:bg-muted/50 first:rounded-t last:rounded-b"
              >
                <span className="font-medium text-foreground">
                  {company.ticker}
                </span>
                <span className="truncate text-muted-foreground">
                  {company.name}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Alert bell */}
      <button
        onClick={() => router.push("/alerts")}
        className="relative ml-4 rounded p-2 text-muted-foreground transition-colors duration-200 hover:bg-muted/30 hover:text-foreground"
        aria-label="Alerts"
      >
        <Bell className="h-4 w-4" />
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-danger px-1 text-[10px] font-medium text-white">
            {unreadCount > 99 ? "99+" : unreadCount}
          </span>
        )}
      </button>
    </header>
  );
}
