"use client";

import React from "react";
import {
  XCircle,
  AlertTriangle,
  ServerCrash,
  Clock,
  FileQuestion,
  RotateCcw,
  X,
  HelpCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ErrorEnvelope } from "@/lib/types";

interface ErrorAlertProps {
  error: ErrorEnvelope;
  onRetry?: () => void;
  onDismiss?: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  error,
  onRetry,
  onDismiss,
}) => {
  // Map specific error codes to intuitive titles & icons
  const getErrorStyle = (code: string) => {
    switch (code) {
      case "BACKEND_UNREACHABLE":
      case "MODEL_NOT_READY":
      case "PROVIDER_UNAVAILABLE":
        return {
          icon: <ServerCrash className="w-5 h-5 text-rose-600 flex-shrink-0" />,
          title: "Service Temporarily Unavailable",
          bg: "bg-rose-50 border-rose-200 text-rose-900",
          btnColor: "bg-rose-600 hover:bg-rose-700 text-white",
        };
      case "ANALYSIS_TIMEOUT":
        return {
          icon: <Clock className="w-5 h-5 text-amber-600 flex-shrink-0" />,
          title: "Analysis Timed Out",
          bg: "bg-amber-50 border-amber-200 text-amber-900",
          btnColor: "bg-amber-600 hover:bg-amber-700 text-white",
        };
      case "IMAGE_TOO_LARGE":
      case "IMAGE_TOO_SMALL":
      case "INVALID_IMAGE_TYPE":
      case "CORRUPTED_IMAGE":
        return {
          icon: <FileQuestion className="w-5 h-5 text-orange-600 flex-shrink-0" />,
          title: "Image Upload Requirement",
          bg: "bg-orange-50 border-orange-200 text-orange-900",
          btnColor: "bg-orange-600 hover:bg-orange-700 text-white",
        };
      case "REQUEST_CANCELLED":
        return {
          icon: <AlertTriangle className="w-5 h-5 text-slate-500 flex-shrink-0" />,
          title: "Request Cancelled",
          bg: "bg-slate-50 border-slate-200 text-slate-800",
          btnColor: "bg-slate-700 hover:bg-slate-800 text-white",
        };
      default:
        return {
          icon: <XCircle className="w-5 h-5 text-rose-600 flex-shrink-0" />,
          title: "Analysis Error",
          bg: "bg-rose-50 border-rose-200 text-rose-900",
          btnColor: "bg-rose-600 hover:bg-rose-700 text-white",
        };
    }
  };

  const style = getErrorStyle(error.code);

  return (
    <div className={`rounded-2xl border p-4 sm:p-5 shadow-sm transition-all ${style.bg}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="mt-0.5">{style.icon}</div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <h4 className="font-semibold text-sm sm:text-base tracking-tight">
                {style.title}
              </h4>
              <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-black/5 opacity-70">
                {error.code}
              </span>
            </div>
            <p className="text-sm opacity-90 leading-relaxed">{error.message}</p>

            {error.hint && (
              <div className="flex items-center gap-1.5 text-xs opacity-80 pt-1">
                <HelpCircle className="w-3.5 h-3.5 flex-shrink-0" />
                <span>{error.hint}</span>
              </div>
            )}
          </div>
        </div>

        {onDismiss && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={onDismiss}
            className="h-8 w-8 text-current opacity-60 hover:opacity-100 hover:bg-black/5 -mt-1 -mr-1"
          >
            <X className="w-4 h-4" />
          </Button>
        )}
      </div>

      {onRetry && (
        <div className="mt-4 pt-3 border-t border-black/5 flex items-center justify-end gap-2">
          <Button
            type="button"
            size="sm"
            onClick={onRetry}
            className={`shadow-sm h-8 px-3 text-xs font-medium flex items-center gap-1.5 ${style.btnColor}`}
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Try Again</span>
          </Button>
        </div>
      )}
    </div>
  );
};

export default ErrorAlert;

