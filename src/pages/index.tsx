import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

import React from 'react';

// --- STYLES DEFINITION (UNCHANGED) ---
const heroStyles: React.CSSProperties = {
  textAlign: 'center',
  padding: '100px 20px',
  backgroundColor: '#1f2937',
  color: '#f9fafb',
  borderRadius: '10px',
  margin: '30px 0',
  boxShadow: '0 10px 25px rgba(0, 0, 0, 0.2)',
};

const titleStyles: React.CSSProperties = {
  fontSize: '3.5rem', 
  fontWeight: '900', 
  marginBottom: '15px',
  color: '#34d399', 
  letterSpacing: '-0.03em',
};

const subtitleStyles: React.CSSProperties = {
  fontSize: '1.4rem',
  fontWeight: '400',
  marginBottom: '40px',
  maxWidth: '900px',
  margin: '0 auto 40px',
  lineHeight: '1.5',
  color: '#d1d5db',
};

const buttonStyles: React.CSSProperties = {
  padding: '14px 32px',
  fontSize: '1.2rem',
  fontWeight: '700',
  color: '#1f2937', 
  backgroundColor: '#34d399', 
  border: 'none',
  borderRadius: '8px',
  cursor: 'pointer',
  transition: 'background-color 0.3s ease, transform 0.1s ease',
  boxShadow: '0 6px 15px rgba(52, 211, 153, 0.4)',
};

// --- HeroSection Component ---
export const HeroSection: React.FC = () => {
  return (
    <section style={heroStyles}>
      <h1 style={titleStyles}>
        The Future is Physical
        <br />
        AI & Humanoid Robotics Hackathon
      </h1>
      
      <p style={subtitleStyles}>
        The Physical AI & Humanoid Robotics. Shape the curriculum 
        of tomorrow!
      </p>
      
     {/* ✅ Updated Button with External Link */}
<a 
  href="https://v0-ai-quiz-with-certificate.vercel.app/" 
  target="_blank" 
  rel="noopener noreferrer" 
  style={{ textDecoration: 'none' }}
>
  <button style={buttonStyles}>
    Join the Challenge Now
  </button>
</a>


      <p style={{ marginTop: '25px', fontSize: '1rem', color: '#9ca3af' }}>
        Registration closes soon. Don't miss out!
      </p>
    </section>
  );
};

// --- HomepageHeader (Standard Docusaurus Hero - optional) ---
function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
      </div>
    </header>
  );
}

// --- Home Component ---
export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      
      <HeroSection />

      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
