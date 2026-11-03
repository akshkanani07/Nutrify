"use client";

import React, { useRef, useState } from "react";
import {
  UploadCloud,
  Camera,
  Image as ImageIcon,
  RotateCcw,
  Trash2,
  FileCheck,
  Sparkles,
  ArrowRight,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB
export const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
export const MIN_SHORT_SIDE = 64;

interface ImageUploadZoneProps {
  onFileSelect: (file: File) => void;
  onAnalyze: () => void;
  onClear: () => void;
  selectedFile: File | null;
  previewUrl: string | null;
  disabled?: boolean;
  isLoading?: boolean;
  onError: (error: { code: string; message: string; hint?: string }) => void;
}

export const ImageUploadZone: React.FC<ImageUploadZoneProps> = ({
  onFileSelect,
  onAnalyze,
  onClear,
  selectedFile,
  previewUrl,
  disabled = false,
  isLoading = false,
  onError,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [imageMeta, setImageMeta] = useState<{ width: number; height: number } | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const cameraInputRef = useRef<HTMLInputElement | null>(null);

  const handleSampleFood = (name: string, bgHue: number) => {
    if (disabled || isLoading) return;
    try {
      const canvas = document.createElement("canvas");
      canvas.width = 256;
      canvas.height = 256;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        // Draw colorful textured meal representation
        ctx.fillStyle = `hsl(${bgHue}, 65%, 45%)`;
        ctx.fillRect(0, 0, 256, 256);
        ctx.fillStyle = `hsl(${bgHue}, 75%, 65%)`;
        ctx.beginPath();
        ctx.arc(128, 128, 90, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 18px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(name, 128, 134);

        canvas.toBlob((blob) => {
          if (blob) {
            const fileName = `${name.toLowerCase().replace(/\s+/g, "_")}_sample.jpg`;
            const file = new File([blob], fileName, { type: "image/jpeg" });
            onFileSelect(file);
          }
        }, "image/jpeg", 0.9);
      }
    } catch {
      // Graceful fallback
    }
  };

  // Client-side image validation (type, size, resolution, corruption)
  const validateFile = (file: File): Promise<boolean> => {
    return new Promise((resolve) => {
      // 1. File size check
      if (file.size > MAX_SIZE_BYTES) {
        const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
        onError({
          code: "IMAGE_TOO_LARGE",
          message: `The selected image is ${sizeMb} MB, exceeding the 10 MB limit.`,
          hint: "Please select or compress your photo to under 10 MB.",
        });
        resolve(false);
        return;
      }

      // 2. MIME type check
      if (!ALLOWED_TYPES.includes(file.type)) {
        onError({
          code: "INVALID_IMAGE_TYPE",
          message: `Unsupported file format (${file.type || "unknown"}). Only JPG, PNG, and WEBP are accepted.`,
          hint: "If using HEIC/GIF/SVG, please convert to JPEG or PNG before uploading.",
        });
        resolve(false);
        return;
      }

      // 3. Image load & resolution check
      const img = new Image();
      const objectUrl = URL.createObjectURL(file);

      img.onload = () => {
        URL.revokeObjectURL(objectUrl);
        const w = img.width;
        const h = img.height;
        const shortSide = Math.min(w, h);

        if (shortSide < MIN_SHORT_SIDE) {
          onError({
            code: "IMAGE_TOO_SMALL",
            message: `Image resolution (${w}×${h} px) is too low. Minimum dimension must be at least ${MIN_SHORT_SIDE} px.`,
            hint: "Please take or choose a clearer, higher-resolution photo of your food.",
          });
          resolve(false);
          return;
        }

        setImageMeta({ width: w, height: h });
        resolve(true);
      };

      img.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        onError({
          code: "CORRUPTED_IMAGE",
          message: "Could not decode image bytes. The file appears to be corrupted or truncated.",
          hint: "Please choose a different photo or re-take the picture.",
        });
        resolve(false);
      };

      img.src = objectUrl;
    });
  };

  const handleIncomingFile = async (file: File) => {
    const isValid = await validateFile(file);
    if (isValid) {
      onFileSelect(file);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleIncomingFile(file);
    }
    // Reset input value so re-selecting same file triggers change
    e.target.value = "";
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled || isLoading) return;

    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleIncomingFile(file);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="w-full">
      {/* Hidden inputs for gallery and native mobile camera */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={handleInputChange}
        disabled={disabled || isLoading}
      />
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        capture="environment"
        className="hidden"
        onChange={handleInputChange}
        disabled={disabled || isLoading}
      />

      {!previewUrl ? (
        /* ================= EMPTY STATE / DROPZONE ================= */
        <div className="space-y-4">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              if (!disabled && !isLoading) setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => {
              if (!disabled && !isLoading) fileInputRef.current?.click();
            }}
            className={`border-2 border-dashed rounded-2xl p-8 sm:p-12 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
              isDragOver
                ? "border-emerald-500 bg-emerald-50/80 scale-[1.01]"
                : "border-slate-300 hover:border-emerald-500 hover:bg-slate-50/80 bg-white"
            } ${disabled || isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
          >
            <div className="w-16 h-16 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center mb-4 shadow-sm border border-emerald-100">
              <UploadCloud className="w-8 h-8" />
            </div>

            <h3 className="text-lg font-bold text-slate-800 tracking-tight">
              Drag & Drop meal photo here
            </h3>
            <p className="text-sm text-slate-500 mt-1 max-w-sm">
              or click to browse from your device. Supported formats:{" "}
              <span className="font-semibold text-slate-700">JPG, PNG, WEBP</span>
            </p>

            <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-xs text-slate-400 font-medium">
              <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-600">Max 10 MB</span>
              <span>•</span>
              <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-600">Min 64×64 px</span>
              <span>•</span>
              <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-600">Private Offline AI</span>
            </div>
          </div>

          {/* Quick Try Sample Meals */}
          <div className="p-4 bg-white rounded-xl border border-slate-200/80 shadow-sm space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
                Quick Try Sample Meals
              </span>
              <span className="text-[11px] text-slate-400">Click to test instantly</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <button
                type="button"
                onClick={() => handleSampleFood("Gala Apple", 0)}
                disabled={disabled || isLoading}
                className="p-2 text-left bg-slate-50 hover:bg-emerald-50/70 border border-slate-200 hover:border-emerald-300 rounded-lg transition-all text-xs font-medium text-slate-700"
              >
                🍎 Gala Apple
              </button>
              <button
                type="button"
                onClick={() => handleSampleFood("Chicken Biryani", 35)}
                disabled={disabled || isLoading}
                className="p-2 text-left bg-slate-50 hover:bg-emerald-50/70 border border-slate-200 hover:border-emerald-300 rounded-lg transition-all text-xs font-medium text-slate-700"
              >
                🍛 Chicken Biryani
              </button>
              <button
                type="button"
                onClick={() => handleSampleFood("Greek Salad", 140)}
                disabled={disabled || isLoading}
                className="p-2 text-left bg-slate-50 hover:bg-emerald-50/70 border border-slate-200 hover:border-emerald-300 rounded-lg transition-all text-xs font-medium text-slate-700"
              >
                🥗 Greek Salad
              </button>
              <button
                type="button"
                onClick={() => handleSampleFood("Avocado Toast", 95)}
                disabled={disabled || isLoading}
                className="p-2 text-left bg-slate-50 hover:bg-emerald-50/70 border border-slate-200 hover:border-emerald-300 rounded-lg transition-all text-xs font-medium text-slate-700"
              >
                🥑 Avocado Toast
              </button>
            </div>
          </div>

          {/* Dedicated Mobile Camera & Gallery Action Buttons */}
          <div className="grid grid-cols-2 gap-3 sm:hidden">
            <Button
              type="button"
              variant="outline"
              onClick={() => cameraInputRef.current?.click()}
              disabled={disabled || isLoading}
              className="h-12 border-slate-300 bg-white hover:bg-emerald-50 hover:border-emerald-400 text-slate-800 font-medium shadow-sm flex items-center justify-center gap-2"
            >
              <Camera className="w-4 h-4 text-emerald-600" />
              <span>Camera</span>
            </Button>

            <Button
              type="button"
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled || isLoading}
              className="h-12 border-slate-300 bg-white hover:bg-emerald-50 hover:border-emerald-400 text-slate-800 font-medium shadow-sm flex items-center justify-center gap-2"
            >
              <ImageIcon className="w-4 h-4 text-emerald-600" />
              <span>Gallery</span>
            </Button>
          </div>
        </div>
      ) : (
        /* ================= PREVIEW & CONFIRMATION STATE ================= */
        <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm overflow-hidden p-5 sm:p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileCheck className="w-5 h-5 text-emerald-600" />
              <span className="font-semibold text-slate-800 text-sm sm:text-base">
                Ready for Analysis
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled || isLoading}
                className="text-xs text-slate-600 hover:text-slate-900"
              >
                <RotateCcw className="w-3.5 h-3.5 mr-1" />
                Replace
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={onClear}
                disabled={disabled || isLoading}
                className="text-xs text-rose-600 hover:text-rose-700 hover:bg-rose-50"
              >
                <Trash2 className="w-3.5 h-3.5 mr-1" />
                Remove
              </Button>
            </div>
          </div>

          {/* High-Fidelity Image Container */}
          <div className="relative rounded-xl overflow-hidden bg-slate-900 max-h-80 sm:max-h-96 flex items-center justify-center border border-slate-800 shadow-inner">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Meal preview"
              className="w-full max-h-80 sm:max-h-96 object-contain"
            />
            {imageMeta && (
              <div className="absolute bottom-2.5 right-2.5 px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur text-[11px] font-mono text-slate-300">
                {imageMeta.width}×{imageMeta.height} px
              </div>
            )}
          </div>

          {/* File Metadata Pill & Action CTA */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2 border-t border-slate-100">
            {selectedFile && (
              <div className="text-xs text-slate-500 truncate max-w-xs self-start sm:self-center">
                <span className="font-medium text-slate-700">{selectedFile.name}</span>{" "}
                <span className="text-slate-400">({formatFileSize(selectedFile.size)})</span>
              </div>
            )}

            <div className="flex items-center gap-2 w-full sm:w-auto">
              <Button
                type="button"
                onClick={onAnalyze}
                disabled={disabled || isLoading}
                className="w-full sm:w-auto bg-emerald-600 hover:bg-emerald-700 text-white font-semibold shadow-md px-6 py-2.5 h-auto transition-all active:scale-[0.99] flex items-center justify-center gap-2"
              >
                <Sparkles className="w-4 h-4 text-emerald-200" />
                <span>Analyze Meal</span>
                <ArrowRight className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUploadZone;

