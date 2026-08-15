import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lattice — Agent Platform",
  description: "Autonomous agent team with full observability",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-zinc-950 text-zinc-100 antialiased">{children}</body>
    </html>
  );
}
