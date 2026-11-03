import test from "node:test";
import assert from "node:assert/strict";

import {
  parseAnalysisResponse,
  getConfidenceLevel,
} from "../lib/types";
import {
  MAX_SIZE_BYTES,
  ALLOWED_TYPES,
  MIN_SHORT_SIDE,
} from "../components/ImageUploadZone";

test("1. parseAnalysisResponse correctly parses top-level JSON fields", () => {
  const directPayload = {
    food_name: "bananas raw",
    confidence: 0.8525,
    calories: 200.2,
    protein: 2.45,
    carbs: 51.39,
    fat: 0.74,
    fiber: 5.85,
  };

  const parsed = parseAnalysisResponse(directPayload);

  assert.equal(parsed.food_name, "bananas raw");
  assert.equal(parsed.confidence, 0.8525);
  assert.equal(parsed.calories, 200.2);
  assert.equal(parsed.protein, 2.5); // Math.round to 1 decimal
  assert.equal(parsed.carbs, 51.4);
  assert.equal(parsed.fat, 0.7);
  assert.equal(parsed.fiber, 5.9);
  assert.equal(parsed.status, "ok");
});

test("2. parseAnalysisResponse correctly parses nested AnalysisResponse structure", () => {
  const nestedPayload = {
    success: true,
    request_id: "req-12345",
    provider: "ml_model",
    model_version: "v1.0-efficientnet-b0-mlp",
    status: "ok",
    items: [
      {
        food_key: "apples_gala",
        name: "apples gala with skin raw",
        confidence: 0.912,
        grams: 182.0,
        estimate: true,
        match: "fruit",
        nutrition: {
          calories: 95.0,
          protein_g: 0.5,
          carbs_g: 25.1,
          fat_g: 0.3,
          fiber_g: 4.4,
          sugar_g: 19.0,
          sodium_mg: 1.8,
        },
        candidates: [
          { name: "strawberries raw", score: 0.05 },
          { name: "nectarines raw", score: 0.03 },
        ],
      },
    ],
    totals: {
      calories: 95.0,
      protein_g: 0.5,
      carbs_g: 25.1,
      fat_g: 0.3,
      fiber_g: 4.4,
      sugar_g: 19.0,
      sodium_mg: 1.8,
    },
    score: {
      value: 88,
      band: "excellent",
      breakdown: [
        { label: "Protein Quality", points: 15, max: 30 },
        { label: "Fiber Density", points: 28, max: 30 },
        { label: "Caloric Balance", points: 38, max: 40 },
      ],
    },
    advice: ["Great source of pectin fiber."],
  };

  const parsed = parseAnalysisResponse(nestedPayload);

  assert.equal(parsed.food_name, "apples gala with skin raw");
  assert.equal(parsed.confidence, 0.912);
  assert.equal(parsed.calories, 95.0);
  assert.equal(parsed.protein, 0.5);
  assert.equal(parsed.carbs, 25.1);
  assert.equal(parsed.fat, 0.3);
  assert.equal(parsed.fiber, 4.4);
  assert.equal(parsed.serving_size, 182);
  assert.equal(parsed.score?.value, 88);
  assert.equal(parsed.score?.band, "excellent");
  assert.equal(parsed.candidates.length, 2);
  assert.equal(parsed.status, "ok");
});

test("3. getConfidenceLevel classifies confidence correctly", () => {
  // High: >= 80%
  assert.equal(getConfidenceLevel(0.80), "high");
  assert.equal(getConfidenceLevel(0.95), "high");
  assert.equal(getConfidenceLevel(85), "high");

  // Medium: 50% - 80%
  assert.equal(getConfidenceLevel(0.50), "medium");
  assert.equal(getConfidenceLevel(0.79), "medium");
  assert.equal(getConfidenceLevel(65), "medium");

  // Low: < 50%
  assert.equal(getConfidenceLevel(0.49), "low");
  assert.equal(getConfidenceLevel(0.20), "low");
  assert.equal(getConfidenceLevel(35), "low");
});

test("4. Image upload validation constraints meet enterprise specifications", () => {
  assert.equal(MAX_SIZE_BYTES, 10 * 1024 * 1024); // 10 MB limit
  assert.deepEqual(ALLOWED_TYPES, ["image/jpeg", "image/png", "image/webp"]);
  assert.equal(MIN_SHORT_SIDE, 64);
});

