import { Shield, Clock, Brain, Lock, Users, Zap } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const features = [
  {
    icon: Brain,
    title: "AI-Powered Intelligence",
    description: "Advanced AI trained on comprehensive legal databases for accurate, context-aware responses."
  },
  {
    icon: Clock,
    title: "24/7 Availability",
    description: "Get instant legal guidance anytime, anywhere. No appointments, no waiting."
  },
  {
    icon: Shield,
    title: "Reliable & Accurate",
    description: "Built on verified legal sources with continuous updates to ensure accuracy."
  },
  {
    icon: Lock,
    title: "Secure & Confidential",
    description: "Your conversations are encrypted and private. We take your confidentiality seriously."
  },
  {
    icon: Users,
    title: "Easy to Understand",
    description: "Complex legal jargon translated into clear, simple language anyone can understand."
  },
  {
    icon: Zap,
    title: "Instant Responses",
    description: "Get answers in seconds, not days. Skip the research and get straight to solutions."
  }
];

const Features = () => {
  return (
    <section id="features" className="py-20 px-6">
      <div className="container mx-auto">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground">
            Why Choose Our Legal AI?
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Experience the future of legal assistance with cutting-edge AI technology
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <Card 
              key={index} 
              className="border-border hover:border-primary/50 transition-all hover:shadow-card group"
            >
              <CardContent className="p-6 space-y-4">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-primary-light flex items-center justify-center group-hover:scale-110 transition-transform">
                  <feature.icon className="h-6 w-6 text-primary-foreground" />
                </div>
                <h3 className="text-xl font-semibold text-foreground">
                  {feature.title}
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
