import { useMutation, useQuery, useQueryClient, keepPreviousData } from "@tanstack/react-query";
import type {
  AppSettings,
  LimitPeriod,
  LimitStatus,
  Bet,
  BetResult,
  DataSources,
  EarlySeason,
  League,
  Leg,
  LegPrice,
  ModelVsLines,
  Parlay,
  ParlaysResponse,
  Performance,
  ScheduleDay,
  NflPlayers,
  NflWarming,
  NflWeek,
  ReplaceResponse,
  Replay,
  Sides,
  SlateInfo,
  TierConfig,
  TierName,
} from "./types";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} returned ${res.status}`);
  return res.json();
}

export async function send<T>(method: "POST" | "PUT" | "PATCH" | "DELETE", path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail, () => null);
    throw new Error(typeof detail === "string" ? detail : `${path} returned ${res.status}`);
  }
  return res.json();
}

// While the backend is still building the slate, poll until it's ready.
const pollWhileWarming = (q: { state: { data?: { status?: string } } }) =>
  q.state.data?.status === "warming" || q.state.data?.status === "idle" ? 1500 : false;

/** Filters every search shares: side preference, session vetoes, games switched off. */
export interface PoolFilters {
  sides: Sides;
  vetoed: string[];
  excludedGames: string[];
}

function filterQuery({ sides, vetoed, excludedGames }: PoolFilters) {
  const q = new URLSearchParams({ sides });
  vetoed.forEach((v) => q.append("veto", v));
  excludedGames.forEach((g) => q.append("exclude", g));
  return q.toString();
}

export const useTiers = () =>
  useQuery({ queryKey: ["tiers"], queryFn: () => get<Record<TierName, TierConfig>>("/api/tiers"), staleTime: Infinity });

const weekParam = (league: League, week: number | null | undefined) => (league === "nfl" && week ? `&week=${week}` : "");

export const useSlate = (league: League, week?: number | null) =>
  useQuery({
    queryKey: ["slate", league, league === "nfl" ? week ?? null : null],
    queryFn: () => get<SlateInfo>(`/api/slate?league=${league}${weekParam(league, week)}`),
    refetchInterval: pollWhileWarming,
  });

export const useParlays = (league: League, tier: TierName, bankroll: number, filters: PoolFilters, week?: number | null) =>
  useQuery({
    queryKey: ["parlays", league, tier, bankroll, filters, league === "nfl" ? week ?? null : null],
    queryFn: () =>
      get<ParlaysResponse>(`/api/parlays?league=${league}&tier=${tier}&bankroll=${bankroll}&${filterQuery(filters)}${weekParam(league, week)}`),
    placeholderData: keepPreviousData,
    refetchInterval: pollWhileWarming,
  });

export const useLegs = (league: League, filters: PoolFilters, enabled: boolean) =>
  useQuery({
    queryKey: ["legs", league, filters],
    queryFn: () => get<{ status: string; legs: Leg[] }>(`/api/legs?league=${league}&${filterQuery(filters)}`),
    placeholderData: keepPreviousData,
    refetchInterval: pollWhileWarming,
    enabled,
  });

export const useNflPlayers = (week: number | null, enabled: boolean) =>
  useQuery({
    queryKey: ["nfl-players", week],
    queryFn: () => get<NflPlayers>(`/api/nfl/players${week ? `?week=${week}` : ""}`),
    refetchInterval: pollWhileWarming,
    enabled,
  });

/** NFL fields: `week`, and typed `prices` keyed by base leg id. */
interface NflBody {
  league?: League;
  week?: number | null;
  prices?: Record<string, LegPrice>;
}

export const replaceLeg = (body: NflBody & {
  leg_ids: string[];
  veto_id: string;
  tier: TierName;
  sides: Sides;
  vetoed: string[];
  excluded_games: string[];
  bankroll: number;
  name: string;
}) => send<ReplaceResponse>("POST", "/api/parlay/replace", body);

export const scoreParlay = (body: NflBody & { leg_ids: string[]; tier: TierName; bankroll: number; name?: string }) =>
  send<{ parlay: Parlay }>("POST", "/api/parlay/score", body);

export const usePerformance = () =>
  useQuery({ queryKey: ["performance"], queryFn: () => get<Performance>("/api/performance"), staleTime: Infinity });

export const useDataSources = () =>
  useQuery({ queryKey: ["data-sources"], queryFn: () => get<DataSources>("/api/data-sources"), refetchInterval: 60_000 });

export const useModelVsLines = () =>
  useQuery({ queryKey: ["model-vs-lines"], queryFn: () => get<ModelVsLines>("/api/nfl/model-vs-lines"), staleTime: Infinity });

export const useBets = () => useQuery({ queryKey: ["bets"], queryFn: () => get<{ bets: Bet[] }>("/api/bets") });

export function useBetMutations() {
  const qc = useQueryClient();
  const done = () => {
    qc.invalidateQueries({ queryKey: ["bets"] });
    qc.invalidateQueries({ queryKey: ["limits"] });
  };
  return {
    log: useMutation({ mutationFn: (bet: Omit<Bet, "id" | "created_at" | "result" | "settled_at" | "profit" | "bonus"> & { bonus?: boolean; confirm_over_limit?: boolean }) => send<Bet>("POST", "/api/bets", bet), onSuccess: done }),
    settle: useMutation({ mutationFn: ({ id, result }: { id: number; result: BetResult }) => send<Bet>("PATCH", `/api/bets/${id}`, { result }), onSuccess: done }),
    remove: useMutation({ mutationFn: (id: number) => send<{ ok: true }>("DELETE", `/api/bets/${id}`), onSuccess: done }),
  };
}

export const useNflWeek = (week: number | null, enabled: boolean) =>
  useQuery({
    queryKey: ["nfl-week", week],
    queryFn: () => get<NflWeek | NflWarming>(`/api/schedule?league=nfl${week ? `&week=${week}` : ""}`),
    placeholderData: keepPreviousData,
    refetchInterval: pollWhileWarming,
    enabled,
    retry: 4, // 503 while an NFL data job holds the warehouse
    retryDelay: 2500,
  });

export const useSchedule = (league: League, date: string | null) =>
  useQuery({
    queryKey: ["schedule", league, date],
    queryFn: () => get<ScheduleDay>(`/api/schedule?league=${league}${date ? `&date=${date}` : ""}`),
    placeholderData: keepPreviousData,
    retry: 4, // 503 while a data job or slate build holds the warehouse
    retryDelay: 2500,
  });

export const useReplay = (league: League, market: string | null, period: string | null, weeks: number) =>
  useQuery({
    queryKey: ["replay", league, market, period, weeks],
    queryFn: () => {
      const q = new URLSearchParams({ league, weeks: String(weeks) });
      if (market) q.set("market", market);
      if (period) q.set("period", period);
      return get<Replay>(`/api/replay?${q}`);
    },
    placeholderData: keepPreviousData,
    retry: 4, // 503 while a data job holds a warehouse
    retryDelay: 2500,
  });

export const useEarlySeason = (enabled = true) =>
  useQuery({
    queryKey: ["early-season"],
    queryFn: () => get<EarlySeason>("/api/nba/early-season"),
    enabled,
    retry: 4, // 503 while a data job holds the warehouse
    retryDelay: 2500,
  });

export const useLimits = () =>
  useQuery({ queryKey: ["limits"], queryFn: () => get<LimitStatus>("/api/limits"), refetchInterval: 60_000 });

export const useSettings = () => useQuery({ queryKey: ["settings"], queryFn: () => get<AppSettings>("/api/settings") });

export function useSettingsMutations() {
  const qc = useQueryClient();
  const limitsDone = (data: LimitStatus) => qc.setQueryData(["limits"], data);
  return {
    setLimit: useMutation({
      mutationFn: ({ period, amount }: { period: LimitPeriod; amount: number | null }) => send<LimitStatus>("PUT", `/api/limits/${period}`, { amount }),
      onSuccess: limitsDone,
    }),
    cancelPending: useMutation({ mutationFn: (period: LimitPeriod) => send<LimitStatus>("DELETE", `/api/limits/${period}/pending`), onSuccess: limitsDone }),
    pause: useMutation({ mutationFn: (days: number) => send<LimitStatus>("POST", "/api/limits/pause", { days }), onSuccess: limitsDone }),
    setKelly: useMutation({
      mutationFn: (kelly_fraction: number) => send<AppSettings>("PUT", "/api/settings/kelly", { kelly_fraction }),
      onSuccess: (data) => {
        qc.setQueryData(["settings"], data);
        qc.invalidateQueries({ queryKey: ["parlays"] });
        qc.invalidateQueries({ queryKey: ["score"] });
      },
    }),
  };
}
