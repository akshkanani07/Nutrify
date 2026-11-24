"use client";

import React, { useState } from "react";
import {
  Flame,
  Dumbbell,
  Wheat,
  Droplet,
  Leaf,
  Scale,
  Sparkles,
  RotateCcw,
  Copy,
  Check,
  Share2,
  ChevronRight,
  TrendingUp,
  HeartPulse,
  Info,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
} from "recharts";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { FoodPrediction, CandidateFood } from "@/lib/types";

interface NutritionResultViewProps {
  prediction: FoodPrediction;
  onReset: () => void;
  onSelectCandidate?: (candidate: CandidateFood) => void;
}

export const NutritionResultView: React.FC<NutritionResultViewProps> = ({
  prediction,
  onReset,
  onSelectCandidate,
}) => {
  const baseServing = prediction.serving_size || 100;
  const [currentServing, setCurrentServing] = useState<number>(baseServing);
  const [copied, setCopied] = useState<boolean>(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<boolean>(false);
  const [showCorrection, setShowCorrection] = useState<boolean>(false);
  const [submittingFeedback, setSubmittingFeedback] = useState<boolean>(false);

  const handleFeedback = async (isCorrect: boolean, selectedFood?: string) => {
    setSubmittingFeedback(true);
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: prediction.request_id || "web-client",
          food_name: prediction.food_name,
          is_correct: isCorrect,
          selected_food: selectedFood || null,
        }),
      });
      setFeedbackSubmitted(true);
      setShowCorrection(false);
    } catch {
      setFeedbackSubmitted(true);
    } finally {
      setSubmittingFeedback(false);
    }
  };

  // Scaling factor for interactive portion adjustments
  const scale = currentServing / baseServing;

  const scaledCalories = Math.round(prediction.calories * scale * 10) / 10;
  const scaledProtein = Math.round(prediction.protein * scale * 10) / 10;
  const scaledCarbs = Math.round(prediction.carbs * scale * 10) / 10;
  const scaledFat = Math.round(prediction.fat * scale * 10) / 10;
  const scaledFiber = Math.round(prediction.fiber * scale * 10) / 10;
  const scaledSugar = Math.round((prediction.sugar || 0) * scale * 10) / 10;
  const scaledSodium = Math.round((prediction.sodium || 0) * scale * 10) / 10;

  // Calorie calculations (4 kcal/g for Protein & Carbs, 9 kcal/g for Fat)
  const proteinCals = scaledProtein * 4;
  const carbsCals = scaledCarbs * 4;
  const fatCals = scaledFat * 9;
  const totalMacroCals = proteinCals + carbsCals + fatCals || 1;

  const calorieSplitData = [
    { name: "Carbs", value: Math.round((carbsCals / totalMacroCals) * 100), color: "#3b82f6" },
    { name: "Protein", value: Math.round((proteinCals / totalMacroCals) * 100), color: "#10b981" },
    { name: "Fat", value: Math.round((fatCals / totalMacroCals) * 100), color: "#f59e0b" },
  ];

  const macroBarData = [
    { name: "Carbs", grams: scaledCarbs, color: "#3b82f6" },
    { name: "Protein", grams: scaledProtein, color: "#10b981" },
    { name: "Fat", grams: scaledFat, color: "#f59e0b" },
    { name: "Fiber", grams: scaledFiber, color: "#8b5cf6" },
    { name: "Sugar", grams: scaledSugar, color: "#ec4899" },
  ];

  const handleCopySummary = () => {
    const text = `Nutrify AI Nutrition Summary:\n` +
      `Food: ${prediction.food_name.toUpperCase()}\n` +
      `Confidence: ${(prediction.confidence * 100).toFixed(1)}%\n` +
      `Serving: ${currentServing}g\n` +
      `Calories: ${scaledCalories} kcal\n` +
      `Protein: ${scaledProtein}g\n` +
      `Carbohydrates: ${scaledCarbs}g (Fiber: ${scaledFiber}g, Sugar: ${scaledSugar}g)\n` +
      `Fat: ${scaledFat}g\n` +
      `Sodium: ${scaledSodium}mg`;

    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const portionPresets = [50, 100, 150, 200, Math.round(baseServing)].filter(
    (v, i, a) => a.indexOf(v) === i
  ).sort((a, b) => a - b);

  return (
    <div className="space-y-6">
      {/* ================= PRIMARY HERO CARD ================= */}
      <Card className="border-slate-200/90 shadow-md overflow-hidden bg-gradient-to-b from-white to-slate-50/50">
        <div className="p-6 sm:p-8 space-y-6">
          {/* Header row: Food Title, Category, and Confidence Badge */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-xs uppercase tracking-wider font-bold text-emerald-700 bg-emerald-50 border border-emerald-200/80 px-2.5 py-0.5 rounded-full">
                  {prediction.category || "Identified Meal"}
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  ID: #{prediction.request_id?.slice(0, 8) || "ml-ai"}
                </span>
              </div>
              <h2 className="text-2xl sm:text-4xl font-extrabold text-slate-900 capitalize tracking-tight">
                {prediction.food_name}
              </h2>
            </div>

            <div className="self-start sm:self-center">
              <ConfidenceBadge confidence={prediction.confidence} size="lg" />
            </div>
          </div>

          {prediction.confidence < 0.40 && (
            <div className="p-3 bg-amber-50/90 border border-amber-200/90 rounded-lg text-xs text-amber-900 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Info className="w-4 h-4 text-amber-600 flex-shrink-0" />
                <span>
                  The neural network scored this at only <strong>{Math.round(prediction.confidence * 100)}% confidence</strong>. If this is an apple or other meal, scroll down to <strong>Alternative Candidates</strong> to switch or provide feedback.
                </span>
              </div>
              {prediction.candidates && prediction.candidates.length > 0 && (
                <a
                  href="#alternatives"
                  className="inline-flex items-center justify-center px-2.5 py-1 bg-amber-600 hover:bg-amber-700 text-white rounded font-medium text-xs whitespace-nowrap transition-colors self-start sm:self-auto"
                >
                  View Alternatives ↓
                </a>
              )}
            </div>
          )}

          {/* Interactive Portion Scaler */}
          <div className="pt-4 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-50/80 p-4 rounded-xl border border-slate-200/70">
            <div className="flex items-center gap-2 text-sm text-slate-700">
              <Scale className="w-4 h-4 text-emerald-600" />
              <span className="font-semibold">Portion Size:</span>
              <span className="font-mono text-base font-bold text-slate-900">
                {currentServing} grams
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-1.5">
              {portionPresets.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => setCurrentServing(preset)}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all ${
                    currentServing === preset
                      ? "bg-emerald-600 text-white shadow-sm font-semibold"
                      : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-100"
                  }`}
                >
                  {preset}g {preset === baseServing ? "(Default)" : ""}
                </button>
              ))}
            </div>
          </div>

          {/* ================= 5 CORE NUTRITIONAL METRIC CARDS ================= */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {/* Calories */}
            <div className="col-span-2 sm:col-span-1 p-4 rounded-xl bg-gradient-to-br from-amber-500/10 via-amber-500/5 to-transparent border border-amber-200/70 shadow-sm flex flex-col justify-between">
              <div className="flex items-center justify-between text-amber-700">
                <span className="text-xs font-semibold uppercase tracking-wider">Calories</span>
                <Flame className="w-4 h-4 text-amber-600" />
              </div>
              <div className="mt-2">
                <span className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight font-mono">
                  {scaledCalories}
                </span>
                <span className="text-xs text-slate-500 block">kcal</span>
              </div>
            </div>

            {/* Protein */}
            <div className="p-4 rounded-xl bg-gradient-to-br from-emerald-500/10 via-emerald-500/5 to-transparent border border-emerald-200/70 shadow-sm flex flex-col justify-between">
              <div className="flex items-center justify-between text-emerald-700">
                <span className="text-xs font-semibold uppercase tracking-wider">Protein</span>
                <Dumbbell className="w-4 h-4 text-emerald-600" />
              </div>
              <div className="mt-2">
                <span className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight font-mono">
                  {scaledProtein}
                </span>
                <span className="text-xs text-slate-500 block">grams</span>
              </div>
            </div>

            {/* Carbohydrates */}
            <div className="p-4 rounded-xl bg-gradient-to-br from-blue-500/10 via-blue-500/5 to-transparent border border-blue-200/70 shadow-sm flex flex-col justify-between">
              <div className="flex items-center justify-between text-blue-700">
                <span className="text-xs font-semibold uppercase tracking-wider">Carbs</span>
                <Wheat className="w-4 h-4 text-blue-600" />
              </div>
              <div className="mt-2">
                <span className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight font-mono">
                  {scaledCarbs}
                </span>
                <span className="text-xs text-slate-500 block">grams</span>
              </div>
            </div>

            {/* Fat */}
            <div className="p-4 rounded-xl bg-gradient-to-br from-amber-500/10 via-amber-500/5 to-transparent border border-amber-200/70 shadow-sm flex flex-col justify-between">
              <div className="flex items-center justify-between text-amber-700">
                <span className="text-xs font-semibold uppercase tracking-wider">Fat</span>
                <Droplet className="w-4 h-4 text-amber-600" />
              </div>
              <div className="mt-2">
                <span className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight font-mono">
                  {scaledFat}
                </span>
                <span className="text-xs text-slate-500 block">grams</span>
              </div>
            </div>

            {/* Dietary Fiber */}
            <div className="p-4 rounded-xl bg-gradient-to-br from-purple-500/10 via-purple-500/5 to-transparent border border-purple-200/70 shadow-sm flex flex-col justify-between">
              <div className="flex items-center justify-between text-purple-700">
                <span className="text-xs font-semibold uppercase tracking-wider">Fiber</span>
                <Leaf className="w-4 h-4 text-purple-600" />
              </div>
              <div className="mt-2">
                <span className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight font-mono">
                  {scaledFiber}
                </span>
                <span className="text-xs text-slate-500 block">grams</span>
              </div>
            </div>
          </div>

          {/* Secondary Details: Sugar, Sodium, Net Carbs */}
          <div className="grid grid-cols-3 gap-2.5 pt-1 text-center">
            <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/60">
              <span className="text-[11px] text-slate-500 block">Sugar</span>
              <strong className="text-sm text-slate-800 font-mono">{scaledSugar} g</strong>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/60">
              <span className="text-[11px] text-slate-500 block">Sodium</span>
              <strong className="text-sm text-slate-800 font-mono">{scaledSodium} mg</strong>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/60">
              <span className="text-[11px] text-slate-500 block">Net Carbs</span>
              <strong className="text-sm text-slate-800 font-mono">
                {Math.max(0, Math.round((scaledCarbs - scaledFiber) * 10) / 10)} g
              </strong>
            </div>
          </div>
        </div>
      </Card>

      {/* ================= CHARTS: MACRO BREAKDOWN & CALORIC RATIOS ================= */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Macronutrient Grams Bar Chart */}
        <Card className="border-slate-200/80 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base sm:text-lg flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-emerald-600" />
              <span>Macronutrient Quantities</span>
            </CardTitle>
            <CardDescription className="text-xs">
              Measured weight in grams per {currentServing}g portion.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-60 pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={macroBarData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#94a3b8" />
                <YAxis tick={{ fontSize: 12 }} stroke="#94a3b8" />
                <Tooltip
                  formatter={(val: number) => [`${val} g`, "Weight"]}
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "none",
                    borderRadius: "8px",
                    color: "#f8fafc",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="grams" radius={[6, 6, 0, 0]}>
                  {macroBarData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Caloric Energy Distribution Donut Chart */}
        <Card className="border-slate-200/80 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base sm:text-lg flex items-center gap-2">
              <Flame className="w-4 h-4 text-amber-500" />
              <span>Caloric Energy Distribution</span>
            </CardTitle>
            <CardDescription className="text-xs">
              Proportion of total energy from each macro.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-60 pt-2 flex items-center justify-between">
            <div className="w-1/2 h-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={calorieSplitData}
                    cx="50%"
                    cy="50%"
                    innerRadius={45}
                    outerRadius={68}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {calorieSplitData.map((entry, index) => (
                      <Cell key={`donut-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(val: number) => [`${val}%`, "Share"]}
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      border: "none",
                      borderRadius: "8px",
                      color: "#f8fafc",
                      fontSize: "12px",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="w-1/2 space-y-2.5 pl-2 text-xs">
              {calorieSplitData.map((item) => (
                <div key={item.name} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full flex-shrink-0"
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="font-medium text-slate-700">{item.name}</span>
                  </div>
                  <span className="font-mono font-bold text-slate-900">{item.value}%</span>
                </div>
              ))}
              <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-400">
                P: 4 kcal/g • C: 4 kcal/g • F: 9 kcal/g
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ================= RUNNER-UP ALTERNATE CANDIDATES ================= */}
      {prediction.candidates && prediction.candidates.length > 0 && (
        <Card id="alternatives" className="border-slate-200/80 shadow-sm scroll-mt-20">
          <CardHeader>
            <CardTitle className="text-base sm:text-lg flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-600" />
              <span>Alternative Detected Candidates</span>
            </CardTitle>
            <CardDescription className="text-xs">
              Neural network runner-up classifications. If the primary prediction is incorrect, select a candidate to update.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {prediction.candidates.slice(0, 4).map((cand, idx) => (
                <div
                  key={idx}
                  className="p-3.5 rounded-xl border border-slate-200 bg-white hover:border-emerald-400 hover:shadow-sm transition-all flex items-center justify-between gap-3 group"
                >
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-slate-400">#{cand.rank}</span>
                      <span className="font-semibold text-sm text-slate-800 capitalize">
                        {cand.food_name}
                      </span>
                    </div>
                    {cand.calories > 0 && (
                      <span className="text-xs text-slate-500 font-mono">
                        {cand.calories} kcal • {cand.protein}g protein
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs font-mono bg-slate-50 text-slate-700">
                      {(cand.confidence * 100).toFixed(1)}%
                    </Badge>
                    {onSelectCandidate && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => onSelectCandidate(cand)}
                        className="h-8 px-2 text-xs text-emerald-600 group-hover:bg-emerald-50"
                      >
                        Select
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* ================= QUALITY SCORE & HEALTH RECOMMENDATIONS ================= */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Health Score */}
        {prediction.score && (
          <Card className="md:col-span-1 border-slate-200/80 shadow-sm flex flex-col justify-between">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <HeartPulse className="w-4 h-4 text-emerald-600" />
                <span>Meal Health Score</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 pt-2">
              <div className="flex items-baseline justify-between">
                <div>
                  <span className="text-4xl font-extrabold text-slate-900 tracking-tight font-mono">
                    {prediction.score.value}
                  </span>
                  <span className="text-slate-400 text-sm font-normal"> / 100</span>
                </div>
                <Badge
                  className={`capitalize px-2.5 py-0.5 text-xs font-bold ${
                    prediction.score.band === "excellent"
                      ? "bg-emerald-100 text-emerald-800 border-emerald-300"
                      : prediction.score.band === "good"
                      ? "bg-blue-100 text-blue-800 border-blue-300"
                      : "bg-amber-100 text-amber-800 border-amber-300"
                  }`}
                >
                  {prediction.score.band}
                </Badge>
              </div>

              {/* Progress sub-bars */}
              <div className="space-y-2 pt-2 border-t border-slate-100">
                {prediction.score.breakdown.map((item, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-600">
                      <span>{item.label}</span>
                      <span className="font-mono">{item.points}/{item.max}</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="bg-emerald-600 h-full rounded-full"
                        style={{ width: `${(item.points / item.max) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Actionable Recommendations */}
        <Card className={`${prediction.score ? "md:col-span-2" : "md:col-span-3"} border-slate-200/80 shadow-sm`}>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Info className="w-4 h-4 text-blue-600" />
              <span>Nutritionist AI Recommendations</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2 space-y-3">
            <ul className="space-y-2.5">
              {prediction.advice.map((line, idx) => (
                <li
                  key={idx}
                  className="flex items-start gap-2.5 text-xs sm:text-sm text-slate-700 bg-slate-50/70 p-2.5 rounded-lg border border-slate-200/50"
                >
                  <span className="text-emerald-600 font-bold mt-0.5">•</span>
                  <span className="leading-relaxed">{line}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* ================= USER FEEDBACK LOOP ================= */}
      <Card className="border-slate-200/90 shadow-sm bg-slate-50/70">
        <CardContent className="p-4 sm:p-5">
          {!feedbackSubmitted ? (
            <div className="space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <span className="text-xs sm:text-sm font-semibold text-slate-800">
                  Was this prediction correct?
                </span>
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleFeedback(true)}
                    disabled={submittingFeedback}
                    className="h-8 px-3 text-xs bg-white text-emerald-700 border-emerald-300 hover:bg-emerald-50"
                  >
                    <Check className="w-3.5 h-3.5 mr-1 text-emerald-600" />
                    <span>Yes, Correct</span>
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setShowCorrection(!showCorrection)}
                    disabled={submittingFeedback}
                    className="h-8 px-3 text-xs bg-white text-rose-700 border-rose-300 hover:bg-rose-50"
                  >
                    <RotateCcw className="w-3.5 h-3.5 mr-1 text-rose-600" />
                    <span>Wrong</span>
                  </Button>
                </div>
              </div>

              {showCorrection && (
                <div className="pt-2 border-t border-slate-200/60 space-y-2">
                  <p className="text-xs text-slate-600">
                    Select the actual food from top predictions to help retrain the model:
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {prediction.candidates && prediction.candidates.length > 0 ? (
                      prediction.candidates.map((cand) => (
                        <button
                          key={cand.rank}
                          type="button"
                          onClick={() => handleFeedback(false, cand.food_name)}
                          className="px-2.5 py-1 text-xs bg-white rounded-md border border-slate-300 text-slate-700 hover:border-emerald-500 hover:text-emerald-700 transition-colors"
                        >
                          {cand.food_name}
                        </button>
                      ))
                    ) : (
                      <button
                        type="button"
                        onClick={() => handleFeedback(false, "Unknown")}
                        className="px-2.5 py-1 text-xs bg-white rounded-md border border-slate-300 text-slate-700"
                      >
                        Food not listed
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-xs text-emerald-800 bg-emerald-50 p-2.5 rounded-md border border-emerald-200">
              <Check className="w-4 h-4 text-emerald-600 flex-shrink-0" />
              <span>Feedback recorded! Thank you for helping improve Nutrition AI.</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ================= ACTIONS BAR ================= */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-4 border-t border-slate-200">
        <Button
          type="button"
          variant="outline"
          onClick={onReset}
          className="w-full sm:w-auto text-slate-700 border-slate-300 hover:bg-slate-50 flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4 text-slate-500" />
          <span>Scan Another Meal</span>
        </Button>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Button
            type="button"
            variant="outline"
            onClick={handleCopySummary}
            className="w-full sm:w-auto text-slate-700 border-slate-300 hover:bg-slate-50 flex items-center gap-1.5"
          >
            {copied ? (
              <>
                <Check className="w-4 h-4 text-emerald-600" />
                <span>Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-4 h-4 text-slate-500" />
                <span>Copy Summary</span>
              </>
            )}
          </Button>

          <Button
            type="button"
            className="w-full sm:w-auto bg-emerald-600 hover:bg-emerald-700 text-white font-medium shadow-sm flex items-center gap-1.5"
            onClick={() => {
              if (navigator.share) {
                navigator.share({
                  title: `Nutrify AI: ${prediction.food_name}`,
                  text: `${prediction.food_name}: ${scaledCalories} kcal, ${scaledProtein}g protein.`,
                }).catch(() => {});
              } else {
                handleCopySummary();
              }
            }}
          >
            <Share2 className="w-4 h-4" />
            <span>Share Result</span>
          </Button>
        </div>
      </div>
    </div>
  );
};

export default NutritionResultView;

