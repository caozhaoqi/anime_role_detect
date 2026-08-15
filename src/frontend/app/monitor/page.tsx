"use client";

import React, { useEffect, useRef, useState } from "react";
import { StatCard, BarRow, LineChart, HealthBadge } from "../components/monitor/charts";

type LatencyStat = { count: number; p50: number; p95: number; max: number; avg: number };
type Metrics = Record<string, any>;

const POLL_MS = 5000;
const HISTORY_MAX = 60;

type HPoint = { t: number; mps: number; genJobs: number; trainJobs: number; restarts: number };

const LATENCY_LABELS: Record<string, string> = {
  "t2i.generate.duration": "生成耗时 (秒)",
  "t2i.train.duration": "训练耗时 (秒)",
};

export default function MonitorPage() {
  const [dark, setDark] = useState(false);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [routes, setRoutes] = useState<any>(null);
  const [services, setServices] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);
  const [history, setHistory] = useState<HPoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const historyRef = useRef<HPoint[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem("monitor-dark");
    if (saved !== null) setDark(saved === "1");
    else if (typeof window !== "undefined")
      setDark(window.matchMedia("(prefers-color-scheme: dark)").matches);
  }, []);

  async function load() {
    try {
      const [m, r, s, h] = await Promise.all([
        fetch("/api/t2i/metrics", { credentials: "include" }).then((x) => x.json()),
        fetch("/api/gateway/routes", { credentials: "include" }).then((x) => x.json()),
        fetch("/api/services", { credentials: "include" }).then((x) => x.json()),
        fetch("/api/health", { credentials: "include" }).then((x) => x.json()),
      ]);
      setMetrics(m || {});
      setRoutes(r || {});
      setServices(s || {});
      setHealth(h || {});
      setError(null);
      setUpdatedAt(Date.now());
      const g = (m && m._gauges) || {};
      const c = (m && m._counters) || {};
      const point: HPoint = {
        t: Date.now(),
        mps: Number(g["t2i.generate.mps_peak_mb"] || 0),
        genJobs: Number(c["t2i.generate.jobs"] || 0),
        trainJobs: Number(c["t2i.train.jobs"] || 0),
        restarts: Number(c["t2i.restarts"] || 0),
      };
      const next = [...historyRef.current, point].slice(-HISTORY_MAX);
      historyRef.current = next;
      setHistory(next);
    } catch (e: any) {
      setError(e?.message || "加载失败");
    }
  }

  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, []);

  const gauges = (metrics && metrics._gauges) || {};
  const counters = (metrics && metrics._counters) || {};
  const latencies: Record<string, LatencyStat> = {};
  if (metrics) {
    for (const [k, v] of Object.entries(metrics)) {
      if (k.startsWith("_")) continue;
      if (v && typeof v === "object" && "p50" in v) latencies[k] = v as LatencyStat;
    }
  }

  const wrapBg = dark ? "bg-gray-950 text-gray-100" : "bg-gray-50 text-gray-900";
  const panelCls = dark ? "bg-gray-900 border-gray-800" : "bg-white border-gray-200";
  const subCls = dark ? "text-gray-400" : "text-gray-500";
  const titleCls = dark ? "text-gray-100" : "text-gray-800";

  const svcList = services?.services ? Object.entries(services.services) : [];
  const routeList = routes?.routes || [];

  return (
    <div className={`min-h-screen ${wrapBg} p-6`}>
      <div className="mx-auto max-w-6xl">
        {/* 顶部栏 */}
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-xl font-semibold">可观测面板 · Observability</h1>
              <p className={`mt-1 text-xs ${subCls}`}>
                聚合 t2i 指标 · 服务注册表 · 下游健康（每 {POLL_MS / 1000}s 自动刷新）
              </p>
            </div>
            {health?.status ? <HealthBadge status={health.status} /> : null}
          </div>
          <div className="flex items-center gap-3">
            {updatedAt ? (
              <span className={`text-xs ${subCls}`}>更新于 {new Date(updatedAt).toLocaleTimeString()}</span>
            ) : null}
            <button
              onClick={load}
              className={`rounded-lg px-3 py-1.5 text-sm ${
                dark ? "bg-gray-800 hover:bg-gray-700 text-gray-200" : "bg-gray-100 hover:bg-gray-200 text-gray-700"
              }`}
            >
              手动刷新
            </button>
            <button
              onClick={() => {
                const v = !dark;
                setDark(v);
                localStorage.setItem("monitor-dark", v ? "1" : "0");
              }}
              className={`rounded-lg px-3 py-1.5 text-sm ${
                dark ? "bg-gray-800 hover:bg-gray-700 text-gray-200" : "bg-gray-100 hover:bg-gray-200 text-gray-700"
              }`}
            >
              {dark ? "亮色" : "暗色"}
            </button>
          </div>
        </div>

        {error ? (
          <div className="mb-4 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700">
            加载出错：{error}（请确认 api-gateway 与 t2i-service 已启动）
          </div>
        ) : null}

        {/* 概览卡片 */}
        <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
          <StatCard dark={dark} title="生成作业" value={counters["t2i.generate.jobs"] || 0} accent="text-blue-500" hint="累计提交" />
          <StatCard dark={dark} title="训练作业" value={counters["t2i.train.jobs"] || 0} accent="text-purple-500" hint="累计提交" />
          <StatCard dark={dark} title="服务重启" value={counters["t2i.restarts"] || 0} accent="text-orange-500" hint="t2i 进程" />
          <StatCard dark={dark} title="Idle 卸载" value={counters["t2i.idle_unloads"] || 0} accent="text-teal-500" hint="权重卸载次数" />
          <StatCard
            dark={dark}
            title="MPS 内存峰值"
            value={Number(gauges["t2i.generate.mps_peak_mb"] || 0).toFixed(0)}
            unit="MB"
            accent="text-green-500"
            hint="生成期峰值"
          />
          <StatCard
            dark={dark}
            title="推理设备"
            value={gauges["t2i.generate.device_is_mps"] ? "MPS" : "CPU"}
            accent="text-indigo-500"
            hint="最近一次"
          />
        </div>

        {/* 耗时分布 + 实时趋势 */}
        <div className="mb-6 grid gap-4 lg:grid-cols-2">
          <section className={`rounded-xl border p-4 ${panelCls}`}>
            <h2 className={`mb-3 text-sm font-semibold ${titleCls}`}>耗时分布（秒）</h2>
            {Object.keys(latencies).length === 0 ? (
              <div className={`text-xs ${subCls}`}>暂无生成/训练样本</div>
            ) : (
              Object.entries(latencies).map(([k, v]) => {
                const maxV = Math.max(v.p50, v.p95, v.max, v.avg) || 1;
                return (
                  <div key={k} className="mb-4">
                    <div className={`mb-1 text-xs font-medium ${titleCls}`}>{LATENCY_LABELS[k] || k}</div>
                    <BarRow dark={dark} label="P50" value={v.p50} max={maxV} unit="s" color="#3b82f6" />
                    <BarRow dark={dark} label="P95" value={v.p95} max={maxV} unit="s" color="#8b5cf6" />
                    <BarRow dark={dark} label="MAX" value={v.max} max={maxV} unit="s" color="#ef4444" />
                    <BarRow dark={dark} label="AVG" value={v.avg} max={maxV} unit="s" color="#10b981" />
                    <div className={`mt-1 text-xs ${subCls}`}>样本数：{v.count}</div>
                  </div>
                );
              })
            )}
          </section>

          <section className={`rounded-xl border p-4 ${panelCls}`}>
            <h2 className={`mb-3 text-sm font-semibold ${titleCls}`}>实时趋势（滚动 {HISTORY_MAX} 点）</h2>
            <div className="space-y-4">
              <div>
                <div className={`mb-1 text-xs ${subCls}`}>MPS 内存峰值 (MB)</div>
                <LineChart data={history.map((p) => p.mps)} color="#10b981" unit="MB" />
              </div>
              <div>
                <div className={`mb-1 text-xs ${subCls}`}>生成作业累计</div>
                <LineChart data={history.map((p) => p.genJobs)} color="#3b82f6" />
              </div>
              <div>
                <div className={`mb-1 text-xs ${subCls}`}>训练作业累计</div>
                <LineChart data={history.map((p) => p.trainJobs)} color="#8b5cf6" />
              </div>
            </div>
          </section>
        </div>

        {/* 服务注册表（feature D） */}
        <section className={`mb-6 rounded-xl border p-4 ${panelCls}`}>
          <h2 className={`mb-1 text-sm font-semibold ${titleCls}`}>服务注册表 · Service Registry</h2>
          <p className={`mb-3 text-xs ${subCls}`}>
            由后端 ServiceRegistry 自注册，网关解析路由的单一事实源（共 {routeList.length} 条规则）
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className={subCls}>
                  <th className="px-2 py-1.5 font-medium">名称</th>
                  <th className="px-2 py-1.5 font-medium">下游</th>
                  <th className="px-2 py-1.5 font-medium">匹配</th>
                  <th className="px-2 py-1.5 font-medium">模板</th>
                </tr>
              </thead>
              <tbody>
                {routeList.map((r: any, i: number) => {
                  const match = r.match_default
                    ? "default"
                    : (r.match_prefix || []).join(", ") || (r.match_exact || []).join(", ");
                  return (
                    <tr key={r.name || i} className={dark ? "border-t border-gray-800" : "border-t border-gray-100"}>
                      <td className="px-2 py-1.5 font-mono text-blue-500">{r.name}</td>
                      <td className="px-2 py-1.5 font-mono">{r.service}</td>
                      <td className="px-2 py-1.5 font-mono text-gray-500">{match}</td>
                      <td className="px-2 py-1.5 font-mono text-gray-500">{r.template}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* 下游健康 */}
        <section className={`rounded-xl border p-4 ${panelCls}`}>
          <h2 className={`mb-3 text-sm font-semibold ${titleCls}`}>下游服务健康</h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {svcList.length === 0 ? (
              <div className={`text-xs ${subCls}`}>暂无数据</div>
            ) : (
              svcList.map(([key, s]: any) => (
                <div
                  key={key}
                  className={`flex items-center justify-between rounded-lg border px-3 py-2 ${
                    dark ? "border-gray-800 bg-gray-800/50" : "border-gray-100 bg-gray-50"
                  }`}
                >
                  <div>
                    <div className="text-sm font-medium">{s.name || key}</div>
                    <div className={`text-xs ${subCls}`}>{s.url}</div>
                  </div>
                  <HealthBadge status={s.status} />
                </div>
              ))
            )}
          </div>
        </section>

        <p className={`mt-6 text-xs ${subCls}`}>
          说明：指标为进程内存快照（服务重启归零），趋势曲线由前端按轮询采样在浏览器内滚动绘制。
        </p>
      </div>
    </div>
  );
}
