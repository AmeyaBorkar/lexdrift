"use client";

import { useState, useEffect, useCallback } from "react";
import { getAlerts, markAlertRead } from "@/lib/api";
import type { Alert } from "@/lib/api";
import {
  AlertTriangle,
  Bell,
  CheckCircle,
  Info,
  ShieldAlert,
} from "lucide-react";

type FilterTab = "all" | "unread" | "critical" | "high";

const SEVERITY_CONFIG: Record<
  string,
  { color: string; bg: string; icon: React.ReactNode }
> = {
  critical: {
    color: "text-red-400",
    bg: "bg-red-400/10",
    icon: <ShieldAlert className="h-4 w-4" />,
  },
  high: {
    color: "text-orange-400",
    bg: "bg-orange-400/10",
    icon: <AlertTriangle className="h-4 w-4" />,
  },
  medium: {
    color: "text-yellow-400",
    bg: "bg-yellow-400/10",
    icon: <Bell className="h-4 w-4" />,
  },
  low: {
    color: "text-blue-400",
    bg: "bg-blue-400/10",
    icon: <Info className="h-4 w-4" />,
  },
};

function getSeverityConfig(severity: string) {
  return (
    SEVERITY_CONFIG[severity?.toLowerCase()] ?? {
      color: "text-muted-foreground",
      bg: "bg-muted",
      icon: <Bell className="h-4 w-4" />,
    }
  );
}

function formatTimestamp(ts: string) {
  if (!ts) return "";
  try {
    const d = new Date(ts);
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return ts;
  }
}

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<FilterTab>("all");

  const fetchAlerts = useCallback(async () => {
    setLoading(true);
    try {
      const unreadParam = activeTab === "unread" ? true : undefined;
      const res = await getAlerts(unreadParam);
      setAlerts(res ?? []);
    } catch (err) {
      console.error("Failed to fetch alerts:", err);
      setAlerts([]);
    } finally {
      setLoading(false);
    }
  }, [activeTab]);

  useEffect(() => {
    fetchAlerts();
  }, [fetchAlerts]);

  async function handleMarkRead(alertId: number) {
    try {
      await markAlertRead(alertId);
      setAlerts((prev) =>
        prev.map((a) => (a.id === alertId ? { ...a, read: true } : a))
      );
    } catch (err) {
      console.error("Failed to mark alert as read:", err);
    }
  }

  const filteredAlerts = alerts.filter((a) => {
    if (activeTab === "unread") return !a.read;
    if (activeTab === "critical")
      return a.severity?.toLowerCase() === "critical";
    if (activeTab === "high") return a.severity?.toLowerCase() === "high";
    return true;
  });

  const unreadCount = alerts.filter((a) => !a.read).length;

  const tabs: { key: FilterTab; label: string; count?: number }[] = [
    { key: "all", label: "All", count: alerts.length },
    { key: "unread", label: "Unread", count: unreadCount },
    { key: "critical", label: "Critical" },
    { key: "high", label: "High" },
  ];

  return (
    <div className="mx-auto max-w-4xl space-y-8">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-foreground">Alerts</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Drift anomalies and filing notifications
        </p>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-1 rounded-lg border border-border bg-card p-1 w-fit">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              activeTab === tab.key
                ? "bg-muted text-accent"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.label}
            {tab.count != null && (
              <span
                className={`rounded-full px-1.5 py-0.5 text-[10px] font-mono ${
                  activeTab === tab.key
                    ? "bg-accent/10 text-accent"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Alert List */}
      <div className="space-y-3">
        {loading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="rounded-lg border border-border bg-card p-5 animate-pulse"
            >
              <div className="flex items-start gap-3">
                <div className="h-8 w-8 rounded bg-muted" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 w-32 rounded bg-muted" />
                  <div className="h-3 w-full rounded bg-muted" />
                  <div className="h-3 w-48 rounded bg-muted" />
                </div>
              </div>
            </div>
          ))
        ) : filteredAlerts.length === 0 ? (
          <div className="rounded-lg border border-border bg-card p-12 text-center">
            <CheckCircle className="h-8 w-8 text-muted-foreground/50 mx-auto mb-3" />
            <p className="text-sm text-muted-foreground">
              {activeTab === "unread"
                ? "No unread alerts."
                : "No alerts found."}
            </p>
          </div>
        ) : (
          filteredAlerts.map((alert) => {
            const config = getSeverityConfig(alert.severity);
            return (
              <div
                key={alert.id}
                onClick={() => !alert.read && handleMarkRead(alert.id)}
                className={`rounded-lg border border-border bg-card p-5 transition-all ${
                  alert.read
                    ? "opacity-60"
                    : "cursor-pointer hover:border-accent/20"
                }`}
              >
                <div className="flex items-start gap-4">
                  <div
                    className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${config.bg} ${config.color}`}
                  >
                    {config.icon}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider ${config.bg} ${config.color}`}
                      >
                        {alert.severity}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {alert.alert_type}
                      </span>
                      {alert.ticker && (
                        <span className="rounded bg-accent/10 px-1.5 py-0.5 text-[10px] font-mono font-medium text-accent">
                          {alert.ticker}
                        </span>
                      )}
                      {!alert.read && (
                        <span className="h-1.5 w-1.5 rounded-full bg-accent" />
                      )}
                    </div>
                    <p className="text-sm text-foreground leading-relaxed">
                      {alert.message}
                    </p>
                    <p className="text-xs text-muted-foreground mt-2">
                      {formatTimestamp(alert.created_at)}
                    </p>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
