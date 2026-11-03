"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, Cpu, Search, CheckCircle2, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ProcessingAnimationProps {
  previewUrl: string | null;
  onCancel?: () => void;
}

interface Step {
  id: number;
  label: string;
  duration: number; // ms
}

const STEPS: Step[] = [
  { id: 1, label: "Decoding & validating image quality...", duration: 250 },
  { id: 2, label: "Preprocessing & letterbox normalizer (224×224)...", duration: 300 },
  { id: 3, label: "Running EfficientNet-B0 neural classification...", duration: 400 },
  { id: 4, label: "Retrieving USDA nutritional profile & macros...", duration: 250 },
];

export const ProcessingAnimation: React.FC<ProcessingAnimationProps> = ({
  previewUrl,
  onCancel,
}) => {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [progress, setProgress] = useState<number>(12);
  const [secondsRemaining, setSecondsRemaining] = useState<string>("~1.0s");

  useEffect(() => {
    const totalTime = STEPS.reduce((acc, s) => acc + s.duration, 0);
    const startTime = Date.now();

    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const pct = Math.min(95, Math.round((elapsed / totalTime) * 95));
      setProgress(pct);

      if (elapsed < 300) {
        setCurrentStep(1);
        setSecondsRemaining("~1.2s");
      } else if (elapsed < 650) {
        setCurrentStep(2);
        setSecondsRemaining("~0.8s");
      } else if (elapsed < 1050) {
        setCurrentStep(3);
        setSecondsRemaining("~0.4s");
      } else {
        setCurrentStep(4);
        setSecondsRemaining("< 0.2s");
      }
    }, 60);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full bg-white rounded-2xl border border-slate-200/80 shadow-lg overflow-hidden p-6 sm:p-8 space-y-6">
      {/* Scanner Visual Container */}
      <div className="relative w-full max-w-md mx-auto aspect-square max-h-72 rounded-xl overflow-hidden bg-slate-900 border border-slate-800 shadow-inner flex items-center justify-center">
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="Scanning food"
            className="w-full h-full object-contain opacity-80"
          />
        ) : (
          <div className="flex flex-col items-center gap-3 text-slate-500">
            <Cpu className="w-12 h-12 animate-pulse text-emerald-400" />
            <span className="text-xs">Processing Visual Input</span>
          </div>
        )}

        {/* Laser Scanning Beam */}
        <motion.div
          className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-emerald-400 to-transparent shadow-[0_0_15px_3px_rgba(52,211,153,0.8)] z-10"
          animate={{
            top: ["0%", "100%", "0%"],
          }}
          transition={{
            duration: 2.2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Grid Overlay Texture */}
        <div className="absolute inset-0 bg-[radial-gradient(#10b981_1px,transparent_1px)] [background-size:16px_16px] opacity-20 pointer-events-none" />

        {/* Corner Targeting Brackets */}
        <div className="absolute top-3 left-3 w-4 h-4 border-t-2 border-l-2 border-emerald-400" />
        <div className="absolute top-3 right-3 w-4 h-4 border-t-2 border-r-2 border-emerald-400" />
        <div className="absolute bottom-3 left-3 w-4 h-4 border-b-2 border-l-2 border-emerald-400" />
        <div className="absolute bottom-3 right-3 w-4 h-4 border-b-2 border-r-2 border-emerald-400" />

        {/* Neural AI Badge */}
        <div className="absolute bottom-3 inset-x-0 flex justify-center">
          <div className="px-3 py-1 rounded-full bg-slate-950/80 backdrop-blur border border-emerald-500/40 text-[11px] font-mono text-emerald-300 flex items-center gap-1.5 shadow-md">
            <Sparkles className="w-3 h-3 text-emerald-400 animate-spin" />
            <span>EfficientNet-B0 Active</span>
          </div>
        </div>
      </div>

      {/* Status & Progress Information */}
      <div className="max-w-md mx-auto space-y-4 text-center">
        <div>
          <h3 className="text-xl font-bold text-slate-900 tracking-tight flex items-center justify-center gap-2">
            <span>Analyzing your food...</span>
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Running local machine learning inference • Est. wait time:{" "}
            <span className="font-semibold text-emerald-600 font-mono">{secondsRemaining}</span>
          </p>
        </div>

        {/* Linear Progress Bar */}
        <div className="space-y-1.5">
          <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden border border-slate-200">
            <motion.div
              className="bg-emerald-600 h-full rounded-full"
              style={{ width: `${progress}%` }}
              transition={{ ease: "easeOut" }}
            />
          </div>
          <div className="flex justify-between text-[11px] font-mono text-slate-400 px-0.5">
            <span>Progress: {progress}%</span>
            <span>100% Offline Model</span>
          </div>
        </div>

        {/* Step Checklist */}
        <div className="space-y-2 text-left bg-slate-50 rounded-xl p-3.5 border border-slate-200/70">
          {STEPS.map((step) => {
            const isDone = currentStep > step.id;
            const isCurrent = currentStep === step.id;
            return (
              <div
                key={step.id}
                className={`flex items-center gap-2.5 text-xs transition-colors ${
                  isDone
                    ? "text-emerald-700 font-medium"
                    : isCurrent
                    ? "text-slate-900 font-semibold"
                    : "text-slate-400"
                }`}
              >
                {isDone ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0" />
                ) : isCurrent ? (
                  <div className="w-3.5 h-3.5 rounded-full border-2 border-emerald-600 border-t-transparent animate-spin flex-shrink-0" />
                ) : (
                  <div className="w-3.5 h-3.5 rounded-full border border-slate-300 flex-shrink-0" />
                )}
                <span className="truncate">{step.label}</span>
              </div>
            );
          })}
        </div>

        {/* Cancellation Button */}
        {onCancel && (
          <div className="pt-1">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onCancel}
              className="text-xs text-slate-600 hover:text-rose-600 hover:bg-rose-50 border-slate-300"
            >
              <X className="w-3.5 h-3.5 mr-1" />
              Cancel Analysis
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ProcessingAnimation;

