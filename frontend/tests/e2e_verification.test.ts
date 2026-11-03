import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

import { parseAnalysisResponse, getConfidenceLevel } from "../lib/types";

test("E2E Workflow: End-to-end payload parsing and confidence UX evaluation", () => {
  // Simulate complete payload returned by FastAPI backend
  const rawBackendOutput = {
    success: true,
    request_id: "test-e2e-run-001",
    provider: "ml_model",
    model_version: "v1.0-efficientnet-b0-mlp",
    status: "ok",
    food_name: "bananas raw",
    confidence: 0.8525,
    calories: 200.2,
    protein: 2.45,
    carbs: 51.39,
    fat: 0.74,
    fiber: 5.85,
    category: "fruit",
    serving_weight_g: 225.0,
    items: [
      {
        food_key: "bananas_raw",
        name: "bananas raw",
        confidence: 0.8525,
        grams: 225.0,
        estimate: true,
        match: "curated",
        nutrition: {
          calories: 200.2,
          protein_g: 2.45,
          carbs_g: 51.39,
          fat_g: 0.74,
          fiber_g: 5.85,
          sugar_g: 27.5,
          sodium_mg: 2.2,
        },
        candidates: [
          { food_key: "potato_pancakes", name: "potato pancakes", score: 0.0103 },
          { food_key: "onions_red_raw", name: "onions red raw", score: 0.0072 },
        ],
      },
    ],
    totals: {
      calories: 200.2,
      protein_g: 2.45,
      carbs_g: 51.39,
      fat_g: 0.74,
      fiber_g: 5.85,
      sugar_g: 27.5,
      sodium_mg: 2.2,
    },
    score: {
      value: 85,
      band: "excellent",
      breakdown: [
        { label: "Protein Quality", points: 15.0, max: 30.0 },
        { label: "Fiber Density", points: 30.0, max: 30.0 },
        { label: "Caloric Balance", points: 40.0, max: 40.0 },
      ],
    },
    advice: ["Great source of potassium and soluble dietary fiber."],
  };

  const parsed = parseAnalysisResponse(rawBackendOutput);

  // 1. Food Name & Confidence
  assert.equal(parsed.food_name, "bananas raw");
  assert.equal(parsed.confidence, 0.8525);
  assert.equal(getConfidenceLevel(parsed.confidence), "high");

  // 2. Nutrition Breakdown
  assert.equal(parsed.calories, 200.2);
  assert.equal(parsed.protein, 2.5);
  assert.equal(parsed.carbs, 51.4);
  assert.equal(parsed.fat, 0.7);
  assert.equal(parsed.fiber, 5.9);
  assert.equal(parsed.sugar, 27.5);
  assert.equal(parsed.sodium, 2.2);

  // 3. Serving Size & Health Score
  assert.equal(parsed.serving_size, 225);
  assert.equal(parsed.score?.value, 85);
  assert.equal(parsed.score?.band, "excellent");

  // 4. Candidates list
  assert.equal(parsed.candidates.length, 2);
  assert.equal(parsed.candidates[0].food_name, "potato pancakes");

  // 5. Portions Scaling Simulation (e.g. user selects 100g instead of 225g)
  const scale = 100 / parsed.serving_size;
  const scaledCalories = Math.round(parsed.calories * scale * 10) / 10;
  assert.equal(scaledCalories, 89.0); // 200.2 * (100 / 225) = 88.97 -> 89.0 kcal
});

test("E2E Workflow: Low confidence triggers uncertain status and warning", () => {
  const lowConfPayload = {
    food_name: "spinach raw",
    confidence: 0.38,
    calories: 23.0,
    protein: 2.9,
    carbs: 3.6,
    fat: 0.4,
    fiber: 2.2,
    status: "unsure",
  };

  const parsed = parseAnalysisResponse(lowConfPayload);

  assert.equal(parsed.food_name, "spinach raw");
  assert.equal(parsed.confidence, 0.38);
  assert.equal(getConfidenceLevel(parsed.confidence), "low");
  assert.equal(parsed.status, "unsure");
});

