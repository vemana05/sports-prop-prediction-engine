"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { useDataSources } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { ApiUsage, DataSources, JobStatus } from "@/lib/types";

const when = (iso: string | null) =>
  iso
    ? new Date(iso).toLocaleString("en-US", { timeZone: "America/New_York", weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) + " ET"
    : "—";

function duration(s: number | null) {
  if (s == null) return "—";
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m ${s % 60}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

function Dot({ tone, label }: { tone: "fresh" | "late" | "failed" | "idle"; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <span
        aria-hidden
        className={cn("h-2.5 w-2.5 rounded-full", {
          "bg-safe": tone === "fresh",
          "bg-risky": tone === "late",
          "bg-danger": tone === "failed",
          "bg-muted": tone === "idle",
        })}
      />
      {label}
    </span>
  );
}

function jobTone(j: JobStatus): "fresh" | "late" | "failed" | "idle" {
  if (j.outcome === "failed") return "failed";
  if (j.late) return "late";
  if (j.outcome === "never ran" || j.outcome === "running") return "idle";
  return "fresh";
}

export default function SourcesPage() {
  const { data, error } = useDataSources();
  return (
    <div className="mx-auto max-w-[1100px] px-4 pb-16 pt-8 lg:px-6">
      <h1 className="font-display text-2xl font-bold">Data sources</h1>
      <p className="mt-1 text-sm text-muted">Scheduled jobs, API usage, data freshness and models. Read from logs and the warehouses; makes no API calls.</p>
      {error ? (
        <p className="mt-4 text-danger-ink">Couldn't load status: {(error as Error).message}</p>
      ) : !data ? (
        <div className="mt-6 space-y-3">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : (
        <Body d={data} />
      )}
    </div>
  );
}

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th scope="col" className={cn("px-4 py-2.5 font-medium", right && "text-right")}>{children}</th>;
}

function Td({ children, right, className }: { children: React.ReactNode; right?: boolean; className?: string }) {
  return <td className={cn("px-4 py-2.5 align-top", right && "num text-right", className)}>{children}</td>;
}

function Table({ head, children, min = 640 }: { head: React.ReactNode; children: React.ReactNode; min?: number }) {
  return (
    <div className="mt-4 overflow-x-auto rounded-control border border-border bg-panel">
      <table className="w-full text-left text-sm" style={{ minWidth: min }}>
        <thead className="border-b border-border text-muted"><tr>{head}</tr></thead>
        <tbody className="divide-y divide-border">{children}</tbody>
      </table>
    </div>
  );
}

function Section({ id, title, note, children }: { id: string; title: string; note?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section aria-labelledby={id} className="mt-10">
      <h2 id={id} className="font-display text-xl font-bold">{title}</h2>
      {note && <p className="mt-1 max-w-prose text-sm text-muted">{note}</p>}
      {children}
    </section>
  );
}

function UsageCard({ u }: { u: ApiUsage }) {
  const used = u.used ?? 0;
  const pctOf = (x: number) => `${Math.min(100, (x / u.limit) * 100)}%`;
  const over = u.used != null && u.used >= u.budget;
  return (
    <div className="rounded-control border border-border bg-panel p-4">
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-display text-lg font-semibold">{u.name}</span>
        <span className="num text-sm">
          <span className={cn("text-base font-semibold", over && "text-danger-ink")}>{u.used ?? "—"}</span> used · cap {u.budget} · limit {u.limit}
        </span>
      </div>
      <div className="relative mt-3 h-2.5 rounded-full bg-border" role="img" aria-label={`${used} of ${u.limit} used; our cap is ${u.budget}`}>
        <div className={cn("h-full rounded-full", over ? "bg-danger" : used >= u.budget * 0.8 ? "bg-risky" : "bg-safe")} style={{ width: pctOf(used) }} />
        <div className="absolute -top-1 w-0.5 bg-ink" style={{ left: pctOf(u.budget), height: 18 }} title={`Our cap: ${u.budget}`} />
      </div>
      <p className="mt-2 text-xs text-muted">
        This month. {u.note} Last call {when(u.last_call)}.
      </p>
    </div>
  );
}

function Body({ d }: { d: DataSources }) {
  const q = d.review_queue;
  return (
    <>
      <section aria-label="Alerts" className="mt-5 rounded-slip border border-border bg-panel p-5 sm:p-6">
        {d.flags.length === 0 ? (
          <p><Dot tone="fresh" label="Everything is on schedule." /></p>
        ) : (
          <>
            <p className="font-medium">{d.flags.length} thing{d.flags.length === 1 ? "" : "s"} need{d.flags.length === 1 ? "s" : ""} attention</p>
            <ul className="mt-2 max-w-prose list-disc space-y-1 pl-5 text-sm">
              {d.flags.map((f) => <li key={f}>{f}</li>)}
            </ul>
          </>
        )}
        <p className="mt-3 text-xs text-muted">As of {when(d.as_of)}. Late means more than a day past schedule, or data older than its expected cadence plus a day.</p>
      </section>

      <Section id="jobs-heading" title="Scheduled jobs" note="From each job's newest log in logs/. Error lines have API keys removed.">
        <Table min={860} head={<><Th>Job</Th><Th>Status</Th><Th>Last run</Th><Th right>Duration</Th><Th>Next run</Th><Th>Last error line</Th></>}>
          {d.jobs.map((j) => (
            <tr key={j.key}>
              <th scope="row" className="px-4 py-2.5 align-top font-medium">
                {j.label}
                <div className="text-xs font-normal text-muted">{j.schedule}</div>
              </th>
              <Td>
                <Dot tone={jobTone(j)} label={j.outcome === "success" ? "Success" : j.outcome === "failed" ? "Failed" : j.outcome === "running" ? "Running" : "Never ran"} />
                {j.late && <div className="text-xs font-medium text-risky-ink">Late</div>}
              </Td>
              <Td>{when(j.started_at)}</Td>
              <Td right>{duration(j.duration_s)}</Td>
              <Td>{when(j.next_scheduled)}</Td>
              <Td className="max-w-[340px] break-words font-mono text-xs text-muted">{j.last_error ?? "—"}</Td>
            </tr>
          ))}
        </Table>
      </Section>

      <Section id="usage-heading" title="API usage" note="From the stored usage logs; this page never calls either API.">
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          {d.usage.map((u) => <UsageCard key={u.name} u={u} />)}
        </div>
      </Section>

      <Section id="tables-heading" title="Warehouse tables" note="Last updated is the newest row's load time. Tables with a cadence are written by a scheduled job.">
        {d.warehouses.map((w) => (
          <div key={w.name} className="mt-6">
            <h3 className="font-display text-lg font-semibold">{w.name} <span className="text-sm font-normal text-muted">{w.path}</span></h3>
            {w.error ? (
              <p className="mt-2 text-sm text-danger-ink">{w.error}</p>
            ) : (
              <Table head={<><Th>Table</Th><Th>Status</Th><Th right>Rows</Th><Th>Last updated</Th><Th>Expected</Th></>}>
                {w.tables.map((t) => (
                  <tr key={t.table}>
                    <th scope="row" className="px-4 py-2.5 font-mono text-xs font-medium">{t.table}</th>
                    <Td>{t.cadence_days ? <Dot tone={t.late ? "late" : "fresh"} label={t.late ? "Late" : "Fresh"} /> : <span className="text-muted">On demand</span>}</Td>
                    <Td right>{t.rows.toLocaleString()}</Td>
                    <Td>{t.time_column ? when(t.last_updated) : <span className="text-muted">no load time</span>}</Td>
                    <Td>{t.cadence_days ? (t.cadence_days === 1 ? "Daily" : `Every ${t.cadence_days} days`) : "—"}</Td>
                  </tr>
                ))}
              </Table>
            )}
          </div>
        ))}
      </Section>

      <Section id="models-heading" title="Saved models" note="Files in data/models, newest first. NFL market models are grouped by version (trained-through tag).">
        <Table head={<><Th>League</Th><Th>Model</Th><Th>Version</Th><Th>File date</Th></>}>
          {d.models.map((m) => (
            <tr key={`${m.name}-${m.version}`}>
              <Td>{m.league}</Td>
              <Td>{m.name}{m.markets.length > 0 && <div className="text-xs text-muted">{m.markets.join(", ")}</div>}</Td>
              <Td className="font-mono text-xs">{m.version ?? "—"}</Td>
              <Td>{when(m.modified)}</Td>
            </tr>
          ))}
        </Table>
      </Section>

      <Section
        id="queue-heading"
        title="Unmatched-player review queue"
        note={
          q.total === 0 ? "Nothing waiting." :
            <>{q.total} name{q.total === 1 ? "" : "s"} waiting: {Object.entries(q.by_source).map(([s, n]) => `${s} ${n}`).join(", ")}.{q.truncated && ` Showing the first ${q.rows.length}.`}</>
        }
      >
        {q.rows.length > 0 && (
          <Table min={760} head={<><Th>Source</Th><Th>Name</Th><Th>Reason</Th><Th>Best candidate</Th><Th>Context</Th></>}>
            {q.rows.map((r) => (
              <tr key={`${r.source}-${r.raw_id}-${r.name}`}>
                <Td className="font-mono text-xs">{r.source}</Td>
                <Td>{r.name}</Td>
                <Td className="text-muted">{r.reason}</Td>
                <Td>{r.best_candidate ? <>{r.best_candidate}{r.best_score != null && <span className="num text-muted"> ({r.best_score})</span>}</> : "—"}</Td>
                <Td className="text-muted">{r.context ?? when(r.created_at)}</Td>
              </tr>
            ))}
          </Table>
        )}
      </Section>
    </>
  );
}
