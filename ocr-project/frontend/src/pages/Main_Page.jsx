import React from 'react';
import { ImageIcon } from 'lucide-react';

export default function Main_Page() {
  return (
    // Main container
    <div className="min-h-screen bg-gray-900 flex items-center justify-center">
      <div className="bg-gray-800 rounded-2xl shadow-xl p-6 w-full max-w-xs text-center">
        {/* Header title */}
        <h1 className="text-white text-lg font-semibold mb-4">Doctor’s Handwritten Predictor</h1>

        {/* Upload section */}
        <div className="bg-gray-700 rounded-xl flex flex-col items-center justify-center p-6 mb-4">
          {/* Image upload placeholder icon */}
          <div className="border-2 border-dashed border-gray-500 rounded-xl p-6 mb-3 w-32 h-32 flex items-center justify-center">
            <ImageIcon className="h-12 w-12 text-gray-400" />
          </div>
          <p className="text-green-400 text-sm font-medium mb-2 underline underline-offset-1">Upload an Image</p>
          {/* Predict button */}
          <button className="bg-gradient-to-r from-orange-400 to-pink-500 text-white font-semibold px-6 py-2 rounded-xl shadow-md transition-transform duration-300 hover:scale-105">
            Predict
          </button>
        </div>

        {/* Text prediction and accuracy list */}
        <div className="mt-6 text-left">
          <div className="flex justify-between text-gray-300 text-sm mb-1">
            <span>Text Prediction</span>
            <span>Accuracy</span>
          </div>
          <div className="border-t border-gray-700 my-2" />

          {/* Example results */}
          <div className="flex justify-between text-white text-sm mb-1">
            <span>Parasatukmol</span>
            <span>84%</span>
          </div>
          <div className="flex justify-between text-white text-sm mb-1">
            <span>Shabu</span>
            <span>84%</span>
          </div>
          <div className="flex justify-between text-white text-sm mb-4">
            <span>Marijuana</span>
            <span>84%</span>
          </div>

          <p className="text-gray-500 text-xs text-center mt-4">Analyzed using CNN+BiLSTM</p>
        </div>
      </div>
    </div>
  );
}

/*
  👉 Adding animations:
  - Use Tailwind's built-in animation utilities, e.g., `animate-pulse`, `animate-bounce`, etc.
  - For more control, you can install Framer Motion:
    npm install framer-motion

    Example:
    import { motion } from 'framer-motion';

    Then wrap your divs:
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.8 }}>...</motion.div>
*/
