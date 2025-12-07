import React, { JSX } from 'react';
// We use the 'lucide-react' library for modern icons
import { Brain, Cpu, Zap, LucideIcon } from 'lucide-react';
import clsx from 'clsx';
import Heading from '@theme/Heading';

// Define the Feature type for better TypeScript support
interface FeatureItem {
  title: string;
  icon: LucideIcon;
  description: JSX.Element;
}

// --- 1. FEATURE DATA ---
const FeatureList: FeatureItem[] = [
  {
    title: 'Embodied Intelligence',
    icon: Brain,
    description: (
      <>
        Master the fundamentals of **physical AI systems** that perceive, reason, and act in the real world. 
        Learn how modern humanoid robots integrate vision, language models, and motor control to navigate 
        complex environments and interact naturally with humans.
      </>
    ),
  },
  {
    title: 'Hands-On Robot Programming',
    icon: Cpu,
    description: (
      <>
        Build practical skills with industry-standard frameworks like **ROS2, PyBullet, and MuJoCo**. 
        Program simulated humanoids, implement control algorithms, and deploy AI models for 
        locomotion, manipulation, and `human-robot interaction`.
      </>
    ),
  },
  {
    title: 'Cutting-Edge AI Integration',
    icon: Zap,
    description: (
      <>
        Explore how **foundation models, reinforcement learning, and computer vision** power the next 
        generation of humanoid robots. From Boston Dynamics to Tesla Optimus, understand the AI 
        architectures driving real-world robotics innovation.
      </>
    ),
  },
];

// --- 2. FEATURE ITEM COMPONENT ---
function Feature({ title, icon: Icon, description }: FeatureItem) {
  return (
    // Use Docusaurus grid classes
    <div className={clsx('col col--4 margin-bottom--lg')}>
      <div className="text--center">
        {/* Custom Icon Wrapper for Styling (using custom CSS in index.module.css) */}
        <div className="feature-icon-wrapper">
          <Icon className="feature-icon" strokeWidth={1.5} />
        </div>
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3" className="feature-title">{title}</Heading>
        <p className="feature-description">{description}</p>
      </div>
    </div>
  );
}

// --- 3. MAIN EXPORT COMPONENT ---
export default function HomepageFeatures(): JSX.Element {
  return (
    <section className="homepage-features-section padding-vert--xl">
      <div className="container">
        {/* Custom Title/Subtitle Section */}
        <div className="text--center margin-bottom--xl">
          <Heading as="h2" className="feature-section-title">
            Why This Course?
          </Heading>
          <p className="feature-section-subtitle">
            Bridge the gap between artificial intelligence and physical robotics. Learn to build 
            intelligent systems that don't just think, but move, sense, and interact with the world.
          </p>
        </div>
        
        {/* Feature Grid */}
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}