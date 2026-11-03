"use client";

import React from "react";
import { CheckCircle2, AlertTriangle, AlertOctagon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { getConfidenceLevel, ConfidenceLevel } from "@/lib/types";

interface ConfidenceBadgeProps {
  confidence: number; // 0.0 to 1.0 or percentage 0 to 100
  showPercentage?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({
  confidence,
  showPercentage = true,
  size = "md",
  className = "",
}) => {
  const level: ConfidenceLevel = getConfidenceLevel(confidence);
  const percentage = Math.round(confidence > 1 ? confidence : confidence * 100);

  const sizeClasses = {
    sm: "px-2.5 py-0.5 text-xs gap-1",
    md: "px-3 py-1 text-xs sm:text-sm gap-1.5 font-medium",
    lg: "px-4 py-1.5 text-sm sm:text-base gap-2 font-semibold",
  };

  const iconSizes = {
    sm: "w-3 h-3",
    md: "w-4 h-4",
    lg: "w-5 h-5",
  };

  if (level === "high") {
    return (
      <Badge
        variant="outline"
        className={`bg-emerald-50 text-emerald-800 border-emerald-300 shadow-sm inline-flex items-center ${sizeClasses[size]} ${className}`}
      >
        <CheckCircle2 className={`${iconSizes[size]} text-emerald-600 flex-shrink-0`} />
        <span>High Confidence</span>
        {showPercentage && (
          <span className="font-mono ml-1 text-emerald-700 bg-emerald-100/70 px-1.5 py-0.5 rounded text-xs">
            {percentage}%
          </span>
        )}
      </Badge>
    );
  }

  if (level === "medium") {
    return (
      <Badge
        variant="outline"
        className={`bg-amber-50 text-amber-800 border-amber-300 shadow-sm inline-flex items-center ${sizeClasses[size]} ${className}`}
      >
        <AlertTriangle className={`${iconSizes[size]} text-amber-600 flex-shrink-0`} />
        <span>Medium Confidence</span>
        {showPercentage && (
          <span className="font-mono ml-1 text-amber-700 bg-amber-100/70 px-1.5 py-0.5 rounded text-xs">
            {percentage}%
          </span>
        )}
      </Badge>
    );
  }

  // Low confidence (< 50%)
  return (
    <Badge
      variant="outline"
      className={`bg-rose-50 text-rose-800 border-rose-300 shadow-sm inline-flex items-center ${sizeClasses[size]} ${className}`}
    >
      <AlertOctagon className={`${iconSizes[size]} text-rose-600 flex-shrink-0`} />
      <span>This prediction is uncertain. Please upload a clearer image.</span>
      {showPercentage && (
        <span className="font-mono ml-1 text-rose-700 bg-rose-100/70 px-1.5 py-0.5 rounded text-xs">
          {percentage}%
        </span>
      )}
    </Badge>
  );
};

export default ConfidenceBadge;

