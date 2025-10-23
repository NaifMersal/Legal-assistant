import { Button } from "@/components/ui/button";
import { Scale } from "lucide-react";

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Scale className="h-6 w-6 text-primary" />
            <span className="text-xl font-bold text-foreground">LegalAI</span>
          </div>
          
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-muted-foreground hover:text-foreground transition-colors">
              Features
            </a>
            <a href="#how-it-works" className="text-muted-foreground hover:text-foreground transition-colors">
              How It Works
            </a>
            <a href="#chat" className="text-muted-foreground hover:text-foreground transition-colors">
              Try Now
            </a>
          </div>
          
          <Button variant="default" className="shadow-md">
            Get Started
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
