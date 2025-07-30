'use client';

import { useEffect, useState } from 'react';

interface BodyWrapperProps {
  children: React.ReactNode;
  className?: string;
}

export default function BodyWrapper({ children, className }: BodyWrapperProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Only render children after mounting to avoid hydration mismatches
  if (!mounted) {
    return <div className={className}>{children}</div>;
  }

  return <>{children}</>;
} 