/**
 * ModelTag — coloured pill for ML / DL / NLP / CV tags
 */
import React from "react";
import { TAG_COLORS } from "../constants/models";

export default function ModelTag({ tag, className = "" }) {
  const colors = TAG_COLORS[tag] || { bg: "#f3f4f6", text: "#374151" };
  return (
    <span
      className={`inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full ${className}`}
      style={{ background: colors.bg, color: colors.text }}
    >
      {tag}
    </span>
  );
}
