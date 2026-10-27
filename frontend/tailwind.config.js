/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#07111f",
        panel: "#0d1a2a",
        line: "#1d2c3e",
        muted: "#8da0b5",
        teal: "#38d9c2",
        blue: "#75aaff"
      }
    }
  },
  plugins: []
};
