/**
 * Hybrid Ensemble AI Engine (Meta-Learning + Biometrics Profiling)
 * Combines Multi-Order Markov, Psychological Biases, and Reaction Time Context
 */
class HybridRPSBrain {
  constructor(playerId = "Player_1") {
    this.moves = ["rock", "paper", "scissors"];
    this.counterMove = { rock: "paper", paper: "scissors", scissors: "rock" };
    this.playerId = playerId;

    this.history = [];
    this.totalRounds = 0;
    this.aiCorrectGuesses = 0;

    // ตัวแปรจับเวลาชีวมิติ (Reaction Time Tracking)
    this.lastRoundTimestamp = Date.now();

    // น้ำหนักของ Sub-Experts (Meta-Predictor Weights)
    this.expertWeights = {
      markovOrder1: 1.0,
      markovOrder2: 1.5,
      psychology: 1.2,
      counterMeta: 0.8,
      latencyBias: 1.0
    };

    // Transition Tables
    this.transitions1 = {}; // 1-gram
    this.transitions2 = {}; // 2-gram
    this.latencyPatterns = { fast: {}, slow: {} }; // < 800ms vs >= 800ms

    this.loadProfile(this.playerId);
  }

  setPlayer(playerId) {
    this.saveProfile();
    this.playerId = playerId;
    this.loadProfile(playerId);
  }

  // --- Sub-Experts แต่ละตัว ---

  // Expert 1: Markov Order-1 (ดู 1 ตาล่าสุด)
  predictOrder1() {
    if (this.history.length < 1) return null;
    const lastMove = this.history[this.history.length - 1].human;
    return this.getBestFromDict(this.transitions1[lastMove]);
  }

  // Expert 2: Markov Order-2 (ดู 2 ตาล่าสุด)
  predictOrder2() {
    if (this.history.length < 2) return null;
    const key = `${this.history[this.history.length - 2].human}->${this.history[this.history.length - 1].human}`;
    return this.getBestFromDict(this.transitions2[key]);
  }

  // Expert 3: Psychology (Win-Stay / Lose-Shift)
  predictPsychology() {
    if (this.history.length < 1) return null;
    const last = this.history[this.history.length - 1];
    if (last.outcome === "human_win") {
      return last.human; // ชนะชอบออกซ้ำ
    } else {
      // แพ้มักจะเปลี่ยนเป็นตัวที่เพิ่งชนะตัวเอง หรือสุ่มเปลี่ยน
      return this.counterMove[last.ai];
    }
  }

  // Expert 4: Counter-Meta (ดักทางผู้เล่นที่พยายามแกล้งซ้อนแผน AI)
  predictCounterMeta() {
    if (this.history.length < 1) return null;
    const last = this.history[this.history.length - 1];
    // ผู้เล่นคิดว่า AI จะออกตัวแก้ทางรอบนี้ ผู้เล่นจึงแกล้งออกตัวมาแก้ทาง AI อีกที
    return this.counterMove[this.counterMove[last.ai]];
  }

  // Expert 5: Latency Bias (พฤติกรรมตามความเร็วในการกด)
  predictLatency(currentLatency) {
    const bucket = currentLatency < 800 ? "fast" : "slow";
    return this.getBestFromDict(this.latencyPatterns[bucket]);
  }

  // --- Meta-Predictor (รวมคะแนนตัดสิน) ---
  predictHumanMove(currentLatency) {
    const votes = { rock: 0, paper: 0, scissors: 0 };

    const experts = [
      { name: "markovOrder1", pred: this.predictOrder1() },
      { name: "markovOrder2", pred: this.predictOrder2() },
      { name: "psychology", pred: this.predictPsychology() },
      { name: "counterMeta", pred: this.predictCounterMeta() },
      { name: "latencyBias", pred: this.predictLatency(currentLatency) }
    ];

    // โหวตคะแนนตามน้ำหนักความแม่นของแต่ละโมเดลย่อย
    for (const exp of experts) {
      if (exp.pred && votes[exp.pred] !== undefined) {
        votes[exp.pred] += this.expertWeights[exp.name];
      }
    }

    // หาตัวที่ได้คะแนนโหวตสูงสุด
    let bestMove = this.moves[Math.floor(Math.random() * 3)];
    let highestScore = -1;
    for (const m of this.moves) {
      if (votes[m] > highestScore) {
        highestScore = votes[m];
        bestMove = m;
      }
    }

    return bestMove;
  }

  getAIMove() {
    const reactionTime = Date.now() - this.lastRoundTimestamp;
    const predictedHumanMove = this.predictHumanMove(reactionTime);
    return {
      aiMove: this.counterMove[predictedHumanMove],
      predictedHumanMove: predictedHumanMove,
      latency: reactionTime
    };
  }

  // บันทึกและปรับค่าน้ำหนักโมเดลแบบ Real-time
  recordRound(humanMove, aiMove) {
    const reactionTime = Date.now() - this.lastRoundTimestamp;
    const outcome = this.evaluateOutcome(humanMove, aiMove);
    this.totalRounds++;

    const isPredicted = (this.counterMove[humanMove] === aiMove);
    if (isPredicted) this.aiCorrectGuesses++;

    // 1. ตรวจสอบว่า Expert ตัวไหนทายถูก แล้วให้รางวัล (Reward) เพิ่มน้ำหนัก
    const candidates = {
      markovOrder1: this.predictOrder1(),
      markovOrder2: this.predictOrder2(),
      psychology: this.predictPsychology(),
      counterMeta: this.predictCounterMeta(),
      latencyBias: this.predictLatency(reactionTime)
    };

    for (const key in candidates) {
      if (candidates[key] === humanMove) {
        this.expertWeights[key] += 0.2; // ตัวไหนแม่น ให้เครดิตเพิ่ม
      } else {
        this.expertWeights[key] = Math.max(0.1, this.expertWeights[key] - 0.05); // ปรับลดลงถ้าทายผิด
      }
    }

    // 2. บันทึกข้อมูลเข้าตารางสถิติ
    if (this.history.length >= 1) {
      const last1 = this.history[this.history.length - 1].human;
      this.transitions1[last1] = this.transitions1[last1] || { rock: 0, paper: 0, scissors: 0 };
      this.transitions1[last1][humanMove]++;
    }

    if (this.history.length >= 2) {
      const lastKey = `${this.history[this.history.length - 2].human}->${this.history[this.history.length - 1].human}`;
      this.transitions2[lastKey] = this.transitions2[lastKey] || { rock: 0, paper: 0, scissors: 0 };
      this.transitions2[lastKey][humanMove]++;
    }

    const bucket = reactionTime < 800 ? "fast" : "slow";
    this.latencyPatterns[bucket][humanMove] = (this.latencyPatterns[bucket][humanMove] || 0) + 1;

    this.history.push({ human: humanMove, ai: aiMove, outcome, latency: reactionTime });
    this.lastRoundTimestamp = Date.now(); // รีเซ็ตเวลาสำหรับรอบต่อไป
    this.saveProfile();

    return {
      outcome,
      isPredicted,
      accuracy: this.getAccuracy(),
      dominantExpert: this.getDominantExpert()
    };
  }

  getDominantExpert() {
    let topExp = "";
    let maxWeight = -1;
    for (const k in this.expertWeights) {
      if (this.expertWeights[k] > maxWeight) {
        maxWeight = this.expertWeights[k];
        topExp = k;
      }
    }
    return topExp;
  }

  getBestFromDict(dict) {
    if (!dict) return null;
    let best = null, max = -1;
    for (const m of this.moves) {
      if (dict[m] > max) {
        max = dict[m];
        best = m;
      }
    }
    return max > 0 ? best : null;
  }

 // คำนวณความแม่นยำเฉพาะ 30 ตาล่าสุด (Rolling Window)
getAccuracy() {
  const currentBatch = this.getCurrentBatch();
  const total = currentBatch.rounds.length;
  if (total === 0) return 0;

  // เอาเฉพาะ 30 ตาล่าสุด (หรือเท่าที่มีถ้ายังไม่ถึง 30)
  const windowSize = Math.min(total, 30);
  const recentRounds = currentBatch.rounds.slice(-windowSize);
  const correct = recentRounds.filter(r => r.isPredicted).length;

  return Math.round((correct / windowSize) * 100);
}

  evaluateOutcome(human, ai) {
    if (human === ai) return "draw";
    if (this.counterMove[human] === ai) return "ai_win";
    return "human_win";
  }

  saveProfile() {
    const payload = {
      transitions1: this.transitions1,
      transitions2: this.transitions2,
      latencyPatterns: this.latencyPatterns,
      expertWeights: this.expertWeights,
      history: this.history,
      totalRounds: this.totalRounds,
      aiCorrectGuesses: this.aiCorrectGuesses
    };
    localStorage.setItem(`ai_arcade_hybrid_${this.playerId}`, JSON.stringify(payload));
  }

  loadProfile(playerId) {
    const data = localStorage.getItem(`ai_arcade_hybrid_${playerId}`);
    if (data) {
      const p = JSON.parse(data);
      this.transitions1 = p.transitions1 || {};
      this.transitions2 = p.transitions2 || {};
      this.latencyPatterns = p.latencyPatterns || { fast: {}, slow: {} };
      this.expertWeights = p.expertWeights || this.expertWeights;
      this.history = p.history || [];
      this.totalRounds = p.totalRounds || 0;
      this.aiCorrectGuesses = p.aiCorrectGuesses || 0;
    } else {
      this.resetMemory();
    }
    this.lastRoundTimestamp = Date.now();
  }

  resetMemory() {
    this.transitions1 = {};
    this.transitions2 = {};
    this.latencyPatterns = { fast: {}, slow: {} };
    this.expertWeights = { markovOrder1: 1.0, markovOrder2: 1.5, psychology: 1.2, counterMeta: 0.8, latencyBias: 1.0 };
    this.history = [];
    this.totalRounds = 0;
    this.aiCorrectGuesses = 0;
    this.lastRoundTimestamp = Date.now();
    localStorage.removeItem(`ai_arcade_hybrid_${this.playerId}`);
  }
}