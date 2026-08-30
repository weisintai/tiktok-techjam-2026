import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TikTok TechJam Shopping Copilot Demo",
  description: "A frontend demo for the hybrid conversational shopping agent.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
