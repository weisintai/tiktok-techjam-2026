export type DemoMessage = {
  role: "shopper" | "copilot";
  text: string;
  turn: number;
};

export type DemoProduct = {
  asin: string;
  title: string;
  category: string;
  price: string;
  rating: string;
  match: number;
  badges: string[];
};

export type DemoScenario = {
  id: string;
  name: string;
  intent: "Buying" | "Browsing" | "Override";
  route: string;
  shopperGoal: string;
  messages: DemoMessage[];
  slots: Record<string, string[]>;
  negatives: string[];
  recommendations: DemoProduct[];
  askAttribute: string;
};

export const metrics = [
  { label: "Hit@10", value: "0.995", caption: "public set" },
  { label: "MRR", value: "0.985", caption: "rank quality" },
  { label: "MTTC", value: "2.97", caption: "turns to conversion" },
  { label: "Token cost", value: "0", caption: "offline default" },
];

export const scenarios: DemoScenario[] = [
  {
    id: "buying",
    name: "High-intent buyer",
    intent: "Buying",
    route: "Exact cards + weighted BM25",
    shopperGoal: "Black running shoes under $80 with no leather.",
    messages: [
      {
        role: "shopper",
        turn: 1,
        text: "I need black running shoes under $80.",
      },
      {
        role: "copilot",
        turn: 1,
        text: "Here are my strongest matches. Which matters most next: material, fit, or specific features?",
      },
      {
        role: "shopper",
        turn: 2,
        text: "No leather, and keep them lightweight.",
      },
      {
        role: "copilot",
        turn: 2,
        text: "I locked the negative material constraint and refreshed unseen candidates.",
      },
    ],
    slots: {
      category: ["running shoes"],
      color: ["black"],
      budget: ["under $80"],
      feature: ["lightweight"],
    },
    negatives: ["leather"],
    askAttribute: "fit or sizing",
    recommendations: [
      {
        asin: "B08RUNNER21",
        title: "Breathable Knit Running Shoe",
        category: "Men > Shoes > Running",
        price: "$64.99",
        rating: "4.6",
        match: 98,
        badges: ["black", "lightweight", "synthetic"],
      },
      {
        asin: "B07TRAIL884",
        title: "Road Pace Athletic Sneaker",
        category: "Women > Shoes > Athletic",
        price: "$72.00",
        rating: "4.5",
        match: 95,
        badges: ["under budget", "no leather hit", "cushioned"],
      },
      {
        asin: "B09CUSHION5",
        title: "Cloudstep Everyday Runner",
        category: "Shoes > Running",
        price: "$58.50",
        rating: "4.4",
        match: 93,
        badges: ["BM25 top", "exact color", "new candidate"],
      },
    ],
  },
  {
    id: "browse",
    name: "Open browsing",
    intent: "Browsing",
    route: "Dense diversity + BM25 fusion",
    shopperGoal: "Something stylish for a summer trip.",
    messages: [
      {
        role: "shopper",
        turn: 1,
        text: "I'm looking for something stylish for a summer trip, but I'm still exploring.",
      },
      {
        role: "copilot",
        turn: 1,
        text: "I widened retrieval across nearby categories and selected a clarifying facet.",
      },
      {
        role: "shopper",
        turn: 2,
        text: "Comfort matters more than brand. I like breathable fabrics.",
      },
      {
        role: "copilot",
        turn: 2,
        text: "Great, I distilled that into comfort and breathable feature signals.",
      },
    ],
    slots: {
      category: ["travel clothing"],
      style: ["stylish"],
      use_case: ["summer trip"],
      feature: ["comfortable", "breathable"],
    },
    negatives: [],
    askAttribute: "color",
    recommendations: [
      {
        asin: "B07COTTON18",
        title: "Linen Blend Resort Shirt",
        category: "Women > Clothing > Tops",
        price: "$36.99",
        rating: "4.7",
        match: 91,
        badges: ["breathable", "travel", "diverse category"],
      },
      {
        asin: "B08SANDAL7",
        title: "Soft Strap Walking Sandal",
        category: "Shoes > Sandals",
        price: "$42.00",
        rating: "4.6",
        match: 89,
        badges: ["comfort", "summer", "cross-category"],
      },
      {
        asin: "B06PACK11",
        title: "Lightweight Day Trip Tote",
        category: "Accessories > Bags",
        price: "$29.95",
        rating: "4.3",
        match: 84,
        badges: ["scenario match", "lightweight", "giftable"],
      },
    ],
  },
  {
    id: "override",
    name: "Intent override",
    intent: "Override",
    route: "Slot rewrite + fresh candidate pool",
    shopperGoal: "Blue instead of black, while preserving budget and use case.",
    messages: [
      {
        role: "shopper",
        turn: 1,
        text: "I'm looking for casual black sneakers under $70.",
      },
      {
        role: "copilot",
        turn: 1,
        text: "I found strong matches and asked for the next constraint.",
      },
      {
        role: "shopper",
        turn: 2,
        text: "Actually, blue instead of black. Keep the casual style.",
      },
      {
        role: "copilot",
        turn: 2,
        text: "Color was rewritten, unrelated slots survived, and shown products were cleared.",
      },
    ],
    slots: {
      category: ["sneakers"],
      color: ["blue"],
      style: ["casual"],
      budget: ["under $70"],
    },
    negatives: ["stale color: black"],
    askAttribute: "material",
    recommendations: [
      {
        asin: "B09BLUE204",
        title: "Blue Canvas Low-Top Sneaker",
        category: "Shoes > Fashion Sneakers",
        price: "$49.99",
        rating: "4.5",
        match: 97,
        badges: ["override hit", "casual", "budget kept"],
      },
      {
        asin: "B07DENIM33",
        title: "Everyday Denim Walking Sneaker",
        category: "Shoes > Sneakers",
        price: "$61.20",
        rating: "4.4",
        match: 94,
        badges: ["blue", "new pool", "style kept"],
      },
      {
        asin: "B08SLIP19",
        title: "Soft Knit Slip-On Sneaker",
        category: "Women > Shoes > Loafers",
        price: "$54.00",
        rating: "4.6",
        match: 90,
        badges: ["casual", "lightweight", "unseen"],
      },
    ],
  },
];

export const pipelineSteps = ["Buying", "Browsing", "Override", "Clarify"];

export const quickPrompts = [
  {
    label: "Find T-shirts",
    value: "I am looking for t-shirts. A key requirement is: 67% Polyester, 33% Cotton.",
  },
  {
    label: "Refine Style",
    value: "For that, what matters is: comfortable fabric; casual wear.",
  },
  {
    label: "Change Color",
    value: "Actually, blue instead of black. Keep the casual style.",
  },
];
