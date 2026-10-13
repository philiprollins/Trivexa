export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <path d="M5 7h22L16 27z" fill="none" stroke="var(--accent)" strokeWidth="3.2" strokeLinejoin="round" />
      <path d="M11.5 12h9L16 20z" fill="var(--accent)" />
    </svg>
  );
}
