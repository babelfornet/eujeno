import { rootVars } from './theme.js'
import Nav from './components/Nav.jsx'
import Hero from './components/Hero.jsx'
import HowItWorks from './components/HowItWorks.jsx'
import WhyP2P from './components/WhyP2P.jsx'
import RunNode from './components/RunNode.jsx'
import UseCases from './components/UseCases.jsx'
import CTA from './components/CTA.jsx'
import Footer from './components/Footer.jsx'

export default function App() {
  return (
    <div
      style={{
        ...rootVars,
        background: 'var(--page-bg)',
        color: 'var(--text)',
        fontFamily: "'Hanken Grotesk',system-ui,sans-serif",
        minHeight: '100vh',
        WebkitFontSmoothing: 'antialiased',
        overflowX: 'hidden',
      }}
    >
      <Nav />
      <Hero />
      <HowItWorks />
      <WhyP2P />
      <RunNode />
      <UseCases />
      <CTA />
      <Footer />
    </div>
  )
}
