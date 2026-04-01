"use client";

import { useState, useEffect, useCallback } from "react";
import { getWatchlists, createWatchlist, addToWatchlist } from "@/lib/api";
import type { Watchlist as WatchlistBase } from "@/lib/api";

// Extend with local tickers tracking (API doesn't return tickers on the watchlist object)
interface Watchlist extends WatchlistBase {
  tickers: string[];
}
import { ChevronRight, Loader2, Plus, X } from "lucide-react";
import { useRouter } from "next/navigation";

export default function WatchlistPage() {
  const router = useRouter();
  const [watchlists, setWatchlists] = useState<Watchlist[]>([]);
  const [loading, setLoading] = useState(true);

  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  const [addingToId, setAddingToId] = useState<number | null>(null);
  const [tickerInput, setTickerInput] = useState("");
  const [addingTicker, setAddingTicker] = useState(false);

  const fetchWatchlists = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getWatchlists();
      setWatchlists((res ?? []).map((wl) => ({ ...wl, tickers: [] })));
    } catch (err) {
      console.error("Failed to fetch watchlists:", err);
      setWatchlists([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWatchlists();
  }, [fetchWatchlists]);

  async function handleCreate() {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      await createWatchlist(newName.trim());
      setNewName("");
      setShowCreateForm(false);
      await fetchWatchlists();
    } catch (err) {
      console.error("Failed to create watchlist:", err);
    } finally {
      setCreating(false);
    }
  }

  async function handleAddTicker(watchlistId: number) {
    if (!tickerInput.trim()) return;
    setAddingTicker(true);
    try {
      const addedTicker = tickerInput.trim().toUpperCase();
      await addToWatchlist(watchlistId, addedTicker);
      setWatchlists((prev) =>
        prev.map((wl) =>
          wl.id === watchlistId
            ? { ...wl, tickers: [...wl.tickers, addedTicker] }
            : wl
        )
      );
      setTickerInput("");
      setAddingToId(null);
    } catch (err) {
      console.error("Failed to add ticker:", err);
    } finally {
      setAddingTicker(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-foreground">
            Watchlists
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Track companies you care about
          </p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2 text-sm text-foreground hover:border-accent/40 transition-colors"
        >
          <Plus className="h-3.5 w-3.5 text-muted-foreground" />
          Create Watchlist
        </button>
      </div>

      {/* Inline Create Form */}
      {showCreateForm && (
        <div className="rounded-lg border border-accent/20 bg-card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-foreground">New Watchlist</h3>
            <button
              onClick={() => {
                setShowCreateForm(false);
                setNewName("");
              }}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="flex gap-3">
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              placeholder="Watchlist name"
              className="flex-1 rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/60 focus:border-accent/40 focus:outline-none transition-colors"
              autoFocus
            />
            <button
              onClick={handleCreate}
              disabled={creating || !newName.trim()}
              className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-foreground hover:bg-accent/90 transition-colors disabled:opacity-50"
            >
              {creating ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                "Create"
              )}
            </button>
          </div>
        </div>
      )}

      {/* Watchlist Cards */}
      <div className="space-y-4">
        {loading ? (
          Array.from({ length: 3 }).map((_, i) => (
            <div
              key={i}
              className="rounded-lg border border-border bg-card p-6 animate-pulse"
            >
              <div className="h-5 w-40 rounded bg-muted mb-4" />
              <div className="flex gap-2">
                {Array.from({ length: 4 }).map((_, j) => (
                  <div
                    key={j}
                    className="h-7 w-16 rounded bg-muted"
                  />
                ))}
              </div>
            </div>
          ))
        ) : watchlists.length === 0 ? (
          <div className="rounded-lg border border-border bg-card p-12 text-center">
            <p className="text-sm text-muted-foreground">
              No watchlists yet. Create one to start tracking companies.
            </p>
          </div>
        ) : (
          watchlists.map((wl) => (
            <div
              key={wl.id}
              className="rounded-lg border border-border bg-card p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-foreground">{wl.name}</h3>
                <button
                  onClick={() =>
                    setAddingToId(addingToId === wl.id ? null : wl.id)
                  }
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-accent transition-colors"
                >
                  <Plus className="h-3 w-3" />
                  Add Company
                </button>
              </div>

              {/* Add Ticker Input */}
              {addingToId === wl.id && (
                <div className="flex gap-2 mb-4">
                  <input
                    type="text"
                    value={tickerInput}
                    onChange={(e) => setTickerInput(e.target.value)}
                    onKeyDown={(e) =>
                      e.key === "Enter" && handleAddTicker(wl.id)
                    }
                    placeholder="Enter ticker (e.g. TSLA)"
                    className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-xs font-mono text-foreground placeholder:text-muted-foreground/60 focus:border-accent/40 focus:outline-none transition-colors"
                    autoFocus
                  />
                  <button
                    onClick={() => handleAddTicker(wl.id)}
                    disabled={addingTicker || !tickerInput.trim()}
                    className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-accent-foreground hover:bg-accent/90 transition-colors disabled:opacity-50"
                  >
                    {addingTicker ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      "Add"
                    )}
                  </button>
                  <button
                    onClick={() => {
                      setAddingToId(null);
                      setTickerInput("");
                    }}
                    className="rounded-md border border-border px-2 py-1.5 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              )}

              {/* Ticker Chips */}
              <div className="flex flex-wrap gap-2">
                {wl.tickers.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No companies added yet.
                  </p>
                ) : (
                  wl.tickers.map((t) => (
                    <button
                      key={t}
                      onClick={() => router.push(`/company/${t}`)}
                      className="group flex items-center gap-1.5 rounded-md border border-border bg-background px-3 py-1.5 text-xs transition-colors hover:border-accent/30"
                    >
                      <span className="font-mono font-medium text-accent">
                        {t}
                      </span>
                      <ChevronRight className="h-3 w-3 text-muted-foreground/50 group-hover:text-accent transition-colors" />
                    </button>
                  ))
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
