import { useMutation, useQuery, useQueryClient, keepPreviousData } from "@tanstack/react-query";
import type {
  Bet,
  BetResult,
  League,
  Leg,
  Parlay,
  ParlaysResponse,
  Performance,
  ScheduleDay,
  NflWeek,
  ReplaceResponse,
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

export async function send<T>(method: "POST" | "PATCH" | "DELETE", path: string, body?: unknown): Promise<T> {
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

export const useSlate = (league: League) =>
  useQuery({
    queryKey: ["slate", league],
    queryFn: () => get<SlateInfo>(`/api/slate?league=${league}`),
    refetchInterval: pollWhileWarming,
  });

export const useParlays = (league: League, tier: TierName, bankroll: number, filters: PoolFilters) =>
  useQuery({
    queryKey: ["parlays", league, tier, bankroll, filters],
    queryFn: () => get<ParlaysResponse>(`/api/parlays?league=${league}&tier=${tier}&bankroll=${bankroll}&${filterQuery(filters)}`),
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

export const replaceLeg = (body: {
  leg_ids: string[];
  veto_id: string;
  tier: TierName;
  sides: Sides;
  vetoed: string[];
  excluded_games: string[];
  bankroll: number;
  name: string;
}) => send<ReplaceResponse>("POST", "/api/parlay/replace", body);

export const scoreParlay = (body: { leg_ids: string[]; tier: TierName; bankroll: number }) =>
  send<{ parlay: Parlay }>("POST", "/api/parlay/score", body);

export const usePerformance = () =>
  useQuery({ queryKey: ["performance"], queryFn: () => get<Performance>("/api/performance"), staleTime: Infinity });

export const useBets = () => useQuery({ queryKey: ["bets"], queryFn: () => get<{ bets: Bet[] }>("/api/bets") });

export function useBetMutations() {
  const qc = useQueryClient();
  const done = () => qc.invalidateQueries({ queryKey: ["bets"] });
  return {
    log: useMutation({ mutationFn: (bet: Omit<Bet, "id" | "created_at" | "result" | "settled_at" | "profit">) => send<Bet>("POST", "/api/bets", bet), onSuccess: done }),
    settle: useMutation({ mutationFn: ({ id, result }: { id: number; result: BetResult }) => send<Bet>("PATCH", `/api/bets/${id}`, { result }), onSuccess: done }),
    remove: useMutation({ mutationFn: (id: number) => send<{ ok: true }>("DELETE", `/api/bets/${id}`), onSuccess: done }),
  };
}

export const useNflWeek = (week: number | null, enabled: boolean) =>
  useQuery({
    queryKey: ["nfl-week", week],
    queryFn: () => get<NflWeek>(`/api/schedule?league=nfl${week ? `&week=${week}` : ""}`),
    placeholderData: keepPreviousData,
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
