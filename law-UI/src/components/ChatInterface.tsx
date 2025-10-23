import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { sendChatMessage, generateSessionId, checkHealth } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: any[];
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "مرحباً! أنا مساعدك في مسألة قانونية. كيف يمكنني مساعدتك اليوم؟"
    }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(() => generateSessionId());
  const [isConnected, setIsConnected] = useState(false);
  const { toast } = useToast();
  const scrollRef = useRef<HTMLDivElement>(null);

  // Check API health on mount
  useEffect(() => {
    checkHealth()
      .then(() => {
        setIsConnected(true);
        toast({
          title: "✓ متصل",
          description: "تم الاتصال بالخادم بنجاح",
        });
      })
      .catch(() => {
        setIsConnected(false);
        toast({
          title: "خطأ في الاتصال",
          description: "فشل الاتصال بالخادم. يرجى التحقق من تشغيل الخادم.",
          variant: "destructive",
        });
      });
  }, [toast]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    
    if (!isConnected) {
      toast({
        title: "خطأ",
        description: "غير متصل بالخادم",
        variant: "destructive",
      });
      return;
    }

    const userMessage = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      // Always use chat endpoint (full RAG with LLM)
      const response = await sendChatMessage(userMessage, sessionId, "rag");
      
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: response.answer,
          sources: response.sources || undefined
        }
      ]);
    } catch (error: any) {
      toast({
        title: "خطأ",
        description: error.message || "فشل في إرسال الرسالة",
        variant: "destructive",
      });
      
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: "عذراً، حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى."
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="w-full h-full">
      <Card className="h-full rounded-none border-0 shadow-none">
        <CardHeader className="border-b border-border/50 bg-gradient-to-r from-primary/5 via-primary/10 to-primary/5 backdrop-blur-sm py-4">
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <img 
                src="/logo-small.png" 
                alt="مسألة قانونية" 
                className="h-12 w-12 object-contain"
              />
              <img 
                src="/mas-ala-qanoniya-text-small.png" 
                alt="مسألة قانونية" 
                className="h-9 object-contain"
              />
            </div>
            <div className="flex items-center gap-2">
              {isConnected ? (
                <div className="flex items-center gap-2 text-xs text-green-600">
                  <div className="w-2 h-2 rounded-full bg-green-600 animate-pulse" />
                  <span>متصل</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-xs text-destructive">
                  <AlertCircle className="w-3 h-3" />
                  <span>غير متصل</span>
                </div>
              )}
            </div>
          </CardTitle>
        </CardHeader>
        
        <CardContent className="p-0 flex flex-col h-[calc(100vh-89px)]">
          <ScrollArea className="flex-1">
            <div ref={scrollRef} className="p-6 space-y-6 min-h-full">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex gap-4 animate-fade-in ${
                    message.role === "user" ? "justify-start" : "justify-end"
                  }`}
                >
                  {message.role === "user" && (
                    <img 
                      src="/user-icon-small.png" 
                      alt="User" 
                      className="w-10 h-10 flex-shrink-0"
                    />
                  )}
                  
                  <div
                    className={`max-w-[75%] rounded-2xl px-5 py-3 shadow-sm transition-all ${
                      message.role === "user"
                        ? "bg-gradient-to-br from-primary to-primary/90 text-primary-foreground rounded-tr-md"
                        : "bg-secondary/80 text-secondary-foreground border border-border/50 rounded-tl-md"
                    }`}
                  >
                    <p className="leading-relaxed text-[15px] text-right" dir="rtl">{message.content}</p>
                  </div>
                  
                  {message.role === "assistant" && (
                    <img 
                      src="/bot-icon-small.png" 
                      alt="Bot" 
                      className="w-10 h-10 flex-shrink-0"
                    />
                  )}
                </div>
              ))}
              
              {isLoading && (
                <div className="flex gap-4 animate-fade-in justify-end">
                  <div className="bg-secondary/80 border border-border/50 rounded-2xl rounded-tl-md px-5 py-4 shadow-sm">
                    <div className="flex gap-1.5">
                      <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" />
                      <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce [animation-delay:0.2s]" />
                      <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce [animation-delay:0.4s]" />
                    </div>
                  </div>
                  <img 
                    src="/bot-icon-small.png" 
                    alt="Bot" 
                    className="w-10 h-10 animate-pulse"
                  />
                </div>
              )}
            </div>
          </ScrollArea>
          
          <div className="border-t border-border/50 bg-background/95 backdrop-blur-sm p-4">
            <div className="flex gap-3 items-end">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="اكتب سؤالك القانوني هنا... (Shift + Enter للسطر الجديد)"
                className="flex-1 min-h-[52px] max-h-32 resize-none rounded-xl border-border/50 focus:border-primary/50 transition-colors"
                disabled={isLoading || !isConnected}
                rows={1}
                dir="rtl"
              />
              <Button
                onClick={handleSend}
                disabled={isLoading || !input.trim() || !isConnected}
                size="icon"
                className="h-[52px] w-[52px] rounded-xl shadow-lg hover:shadow-xl transition-shadow bg-gradient-to-br from-primary to-primary/90 p-0 overflow-hidden"
              >
                <img 
                  src="/send-button-smaller.png" 
                  alt="Send" 
                  className="h-full w-full object-contain p-2"
                />
              </Button>
            </div>
            <p className="text-xs text-muted-foreground/70 mt-3 text-center">
              هذا لأغراض إعلامية فقط ولا يشكل استشارة قانونية
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ChatInterface;
