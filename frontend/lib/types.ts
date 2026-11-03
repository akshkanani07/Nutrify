/**
 * Production Nutrition AI TypeScript Types & Contract Parser.
 */

export interface MacroNutrient {
  name: string;
  grams: number;
  calories: number;
  percentageOfCalories: number;
  color: string;
}

export interface CandidateFood {
  rank: number;
  food_name: string;
  confidence: number;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  category?: string;
}

export interface ScoreBreakdownItem {
  label: string;
  points: number;
  max: number;
}

export interface MealScore {
  value: number;
  band: "excellent" | "good" | "fair" | "poor";
  breakdown: ScoreBreakdownItem[];
}

export interface FoodPrediction {
  food_name: string;
  confidence: number; // 0.0 to 1.0
  calories: number;
  protein: number; // grams
  carbs: number; // grams
  fat: number; // grams
  fiber: number; // grams
  sugar: number; // grams
  sodium: number; // mg
  serving_size: number; // grams (default)
  category?: string;
  candidates: CandidateFood[];
  status: "ok" | "unsure" | "no_food";
  score?: MealScore;
  advice: string[];
  provider?: string;
  model_version?: string;
  request_id?: string;
}

export interface ErrorEnvelope {
  code: string;
  message: string;
  hint?: string;
  retryable?: boolean;
}

export type ConfidenceLevel = "high" | "medium" | "low";

export function getConfidenceLevel(confidence: number): ConfidenceLevel {
  const pct = confidence > 1 ? confidence : confidence * 100;
  if (pct >= 80) return "high";
  if (pct >= 50) return "medium";
  return "low";
}

/**
 * Normalizes backend response into strongly typed FoodPrediction.
 * Handles both top-level JSON fields:
 *   { food_name, confidence, calories, protein, carbs, fat, fiber }
 * and nested AnalysisResponse envelope:
 *   { items: [...], totals: {...}, score: {...} }
 */
export function parseAnalysisResponse(data: any): FoodPrediction {
  const firstItem = Array.isArray(data?.items) && data.items.length > 0 ? data.items[0] : null;
  const totals = data?.totals || {};

  // Extract primary name
  const rawName =
    data?.food_name ||
    firstItem?.name ||
    firstItem?.food_key?.replace(/_/g, " ") ||
    "Unknown Food";

  // Clean formatted name
  const food_name = String(rawName).trim();

  // Extract confidence
  let rawConfidence = data?.confidence;
  if (rawConfidence === undefined || rawConfidence === null) {
    rawConfidence = firstItem?.confidence ?? 0.85;
  }
  const confidence = Number(rawConfidence) > 1 ? Number(rawConfidence) / 100 : Number(rawConfidence);

  // Extract base nutrients
  const calories = Number(data?.calories ?? totals?.calories ?? firstItem?.nutrition?.calories ?? 0);
  const protein = Number(data?.protein ?? totals?.protein_g ?? firstItem?.nutrition?.protein_g ?? 0);
  const carbs = Number(data?.carbs ?? totals?.carbs_g ?? firstItem?.nutrition?.carbs_g ?? 0);
  const fat = Number(data?.fat ?? totals?.fat_g ?? firstItem?.nutrition?.fat_g ?? 0);
  const fiber = Number(data?.fiber ?? totals?.fiber_g ?? firstItem?.nutrition?.fiber_g ?? 0);
  const sugar = Number(data?.sugar ?? totals?.sugar_g ?? firstItem?.nutrition?.sugar_g ?? 0);
  const sodium = Number(data?.sodium ?? totals?.sodium_mg ?? firstItem?.nutrition?.sodium_mg ?? 0);

  // Serving size
  const serving_size = Number(data?.serving_weight_g ?? firstItem?.grams ?? 100);

  // Extract category
  const category = data?.category || firstItem?.match || "Food Item";

  // Parse candidates
  const candidates: CandidateFood[] = [];
  if (Array.isArray(data?.candidates) && data.candidates.length > 0) {
    data.candidates.forEach((cand: any, idx: number) => {
      candidates.push({
        rank: cand.rank || idx + 1,
        food_name: cand.food_name || cand.name || "Candidate",
        confidence: Number(cand.confidence || cand.score || 0),
        calories: Number(cand.calories || 0),
        protein: Number(cand.protein || 0),
        carbs: Number(cand.carbs || 0),
        fat: Number(cand.fat || 0),
        fiber: Number(cand.fiber || 0),
        category: cand.category,
      });
    });
  } else if (Array.isArray(firstItem?.candidates)) {
    firstItem.candidates.forEach((cand: any, idx: number) => {
      candidates.push({
        rank: idx + 2,
        food_name: cand.name || cand.food_key?.replace(/_/g, " ") || "Candidate",
        confidence: Number(cand.score || 0),
        calories: 0,
        protein: 0,
        carbs: 0,
        fat: 0,
        fiber: 0,
      });
    });
  }

  // Parse status
  const status: "ok" | "unsure" | "no_food" =
    data?.status === "unsure" || confidence < 0.5
      ? "unsure"
      : data?.status === "no_food"
      ? "no_food"
      : "ok";

  // Parse score
  let score: MealScore | undefined = undefined;
  if (data?.score) {
    score = {
      value: Number(data.score.value || 75),
      band: data.score.band || (data.score.value >= 80 ? "excellent" : "good"),
      breakdown: Array.isArray(data.score.breakdown) ? data.score.breakdown : [],
    };
  } else {
    // Generate default score from macros
    const protScore = Math.min(30, (protein / 20) * 30);
    const fibScore = Math.min(30, (fiber / 8) * 30);
    const calScore = Math.max(10, 40 - Math.abs(calories - 300) * 0.08);
    const totalVal = Math.round(protScore + fibScore + calScore);
    score = {
      value: Math.min(98, Math.max(20, totalVal)),
      band: totalVal >= 80 ? "excellent" : totalVal >= 65 ? "good" : "fair",
      breakdown: [
        { label: "Protein Quality", points: Math.round(protScore), max: 30 },
        { label: "Fiber Density", points: Math.round(fibScore), max: 30 },
        { label: "Caloric Balance", points: Math.round(calScore), max: 40 },
      ],
    };
  }

  // Parse advice
  const advice: string[] = Array.isArray(data?.advice) && data.advice.length > 0
    ? data.advice
    : [
        fiber >= 3 ? "Rich in dietary fiber, promoting gut microbiome diversity." : "Consider pairing with fresh greens or whole grain for additional fiber.",
        protein >= 10 ? "Solid source of quality protein supporting muscle synthesis." : "Add a lean protein source to achieve balanced satiety.",
        calories > 400 ? "Nutrient-dense portion; ideal for fueling active energy needs." : "Light and healthy portion fitting clean dietary goals.",
      ];

  return {
    food_name,
    confidence: Number(confidence.toFixed(4)),
    calories: Math.round(calories * 10) / 10,
    protein: Math.round(protein * 10) / 10,
    carbs: Math.round(carbs * 10) / 10,
    fat: Math.round(fat * 10) / 10,
    fiber: Math.round(fiber * 10) / 10,
    sugar: Math.round(sugar * 10) / 10,
    sodium: Math.round(sodium * 10) / 10,
    serving_size: Math.round(serving_size),
    category,
    candidates,
    status,
    score,
    advice,
    provider: data?.provider || "ml_model",
    model_version: data?.model_version,
    request_id: data?.request_id,
  };
}

