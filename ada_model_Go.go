package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/xuri/excelize/v2"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

// ParameterLogger for Go
type ParameterLogger struct {
	parameters         map[string]interface{}
	performanceMetrics map[string]map[string]interface{}
	timestamp          time.Time
}

func NewParameterLogger() *ParameterLogger {
	return &ParameterLogger{
		parameters:         make(map[string]interface{}),
		performanceMetrics: make(map[string]map[string]interface{}),
		timestamp:          time.Now(),
	}
}

func (pl *ParameterLogger) LogParameters(modelType string, params map[string]interface{}) {
	pl.parameters[modelType] = params
}

func (pl *ParameterLogger) LogPerformance(modelType string, metrics map[string]interface{}) {
	pl.performanceMetrics[modelType] = metrics
}

func (pl *ParameterLogger) ExportToExcel(filename string) error {
	f := excelize.NewFile()

	// Add parameters sheet
	f.NewSheet("Parameters")
	paramsRow := 1
	f.SetCellValue("Parameters", "A1", "Parameter")
	f.SetCellValue("Parameters", "B1", "Value")

	for key, value := range pl.parameters {
		paramsRow++
		f.SetCellValue("Parameters", fmt.Sprintf("A%d", paramsRow), key)
		f.SetCellValue("Parameters", fmt.Sprintf("B%d", paramsRow), fmt.Sprintf("%v", value))
	}

	// Add performance metrics sheet
	f.NewSheet("Performance")
	perfRow := 1
	f.SetCellValue("Performance", "A1", "Metric")
	f.SetCellValue("Performance", "B1", "Value")

	for model, metrics := range pl.performanceMetrics {
		for metric, value := range metrics {
			perfRow++
			f.SetCellValue("Performance", fmt.Sprintf("A%d", perfRow), fmt.Sprintf("%s_%s", model, metric))
			f.SetCellValue("Performance", fmt.Sprintf("B%d", perfRow), value)
		}
	}

	// Add experiment details sheet
	f.NewSheet("Experiment_Details")
	details := map[string]string{
		"A1": "Item", "B1": "Value",
		"A2": "Experiment Name", "B2": "Contextual Bandit Analysis",
		"A3": "Date", "B3": pl.timestamp.Format("2006-01-02"),
		"A4": "Random Seed", "B4": "123",
		"A5": "Data Files", "B5": "contextual_bandit_train.csv, contextual_bandit_test.csv",
		"A6": "Language", "B6": "Go",
		"A7": "Go Version", "B7": runtime.Version(),
	}

	for cell, value := range details {
		f.SetCellValue("Experiment_Details", cell, value)
	}

	// Set Parameters as active sheet
	f.SetActiveSheet(0)

	// Save file
	if err := f.SaveAs(filename); err != nil {
		return err
	}

	fmt.Printf("Parameters and performance metrics exported to %s\n", filename)
	return nil
}

// LinUCB Model with seeding
type LinUCB struct {
	alpha float64
	A     []*mat.Dense
	b     []*mat.VecDense
	d     int
	k     int
	seed  int64
}

func NewLinUCB(alpha float64, d, k int, seed int64) *LinUCB {
	// Set random seed
	rand.Seed(seed)

	A := make([]*mat.Dense, k)
	b := make([]*mat.VecDense, k)

	for i := 0; i < k; i++ {
		// Initialize A as identity matrix
		identity := make([]float64, d*d)
		for j := 0; j < d; j++ {
			identity[j*d+j] = 1.0
		}
		A[i] = mat.NewDense(d, d, identity)
		b[i] = mat.NewVecDense(d, nil)
	}

	return &LinUCB{
		alpha: alpha,
		A:     A,
		b:     b,
		d:     d,
		k:     k,
		seed:  seed,
	}
}

func (l *LinUCB) SelectArm(context []float64) int {
	maxScore := math.Inf(-1)
	bestArm := 0

	for arm := 0; arm < l.k; arm++ {
		// Calculate theta = A^-1 * b
		var AInv mat.Dense
		err := AInv.Inverse(l.A[arm])
		if err != nil {
			continue
		}

		theta := mat.NewVecDense(l.d, nil)
		theta.MulVec(&AInv, l.b[arm])

		// Convert context to vector
		ctxVec := mat.NewVecDense(l.d, context)

		// Calculate score: context^T * theta + alpha * sqrt(context^T * A^-1 * context)
		score := mat.Dot(ctxVec, theta)

		// Calculate UCB term
		var temp mat.VecDense
		temp.MulVec(&AInv, ctxVec)
		ucbTerm := l.alpha * math.Sqrt(mat.Dot(ctxVec, &temp))

		score += ucbTerm

		if score > maxScore {
			maxScore = score
			bestArm = arm
		}
	}
	return bestArm + 1 // Convert to 1-indexed
}

func (l *LinUCB) Update(arm int, context []float64, reward float64) {
	// Convert to 0-indexed
	armIdx := arm - 1

	// Update A: A = A + context * context^T
	ctxVec := mat.NewVecDense(l.d, context)
	var outer mat.Dense
	outer.Outer(1, ctxVec, ctxVec)
	l.A[armIdx].Add(l.A[armIdx], &outer)

	// Update b: b = b + reward * context
	var rewardVec mat.VecDense
	rewardVec.ScaleVec(reward, ctxVec)
	l.b[armIdx].AddVec(l.b[armIdx], &rewardVec)
}

// EpsilonGreedy Model with seeding
type EpsilonGreedy struct {
	epsilon float64
	counts  []int
	values  []float64
	k       int
	seed    int64
}

func NewEpsilonGreedy(epsilon float64, k int, seed int64) *EpsilonGreedy {
	// Set random seed
	rand.Seed(seed)

	return &EpsilonGreedy{
		epsilon: epsilon,
		counts:  make([]int, k),
		values:  make([]float64, k),
		k:       k,
		seed:    seed,
	}
}

func (e *EpsilonGreedy) SelectArm() int {
	if rand.Float64() < e.epsilon {
		return rand.Intn(e.k) + 1 // 1-indexed
	}

	maxValue := math.Inf(-1)
	bestArm := 0
	for arm := 0; arm < e.k; arm++ {
		if e.values[arm] > maxValue {
			maxValue = e.values[arm]
			bestArm = arm
		}
	}
	return bestArm + 1 // 1-indexed
}

func (e *EpsilonGreedy) Update(arm int, reward float64) {
	armIdx := arm - 1 // Convert to 0-indexed
	e.counts[armIdx]++
	n := float64(e.counts[armIdx])
	// Update rule: new_value = old_value + (reward - old_value) / n
	e.values[armIdx] += (reward - e.values[armIdx]) / n
}

// History tracking
type History struct {
	ChosenActions    []int
	Rewards          []float64
	Regrets          []float64
	CumulativeReward []float64
	CumulativeRegret []float64
	ContextUsed      [][]float64
}

// Performance monitoring
type PerformanceMonitor struct {
	startTime time.Time
	startMem  uint64
}

func NewPerformanceMonitor() *PerformanceMonitor {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return &PerformanceMonitor{
		startTime: time.Now(),
		startMem:  m.Alloc,
	}
}

func (pm *PerformanceMonitor) Stop() (time.Duration, uint64) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	elapsed := time.Since(pm.startTime)
	memoryUsed := m.Alloc - pm.startMem
	return elapsed, memoryUsed
}

func getMemoryUsage() uint64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Alloc
}

// Load CSV data
func loadCSV(filename string) ([][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return records[1:], nil // Skip header
}

// Parse float from record
func parseFloatFromRecord(record []string, index int) float64 {
	val, err := strconv.ParseFloat(record[index], 64)
	if err != nil {
		return 0.0
	}
	return val
}

// Parse context from record (first 10 columns)
func parseContext(record []string) []float64 {
	context := make([]float64, 10)
	for i := 0; i < 10; i++ {
		context[i] = parseFloatFromRecord(record, i)
	}
	return context
}

// Adaptive Decision Analysis
func ada_model_go(trainFile, testFile string, modelType string, logger *ParameterLogger) *History {
	// Load data
	trainData, err := loadCSV(trainFile)
	if err != nil {
		log.Fatal("Error loading training data:", err)
	}

	testData, err := loadCSV(testFile)
	if err != nil {
		log.Fatal("Error loading test data:", err)
	}

	// Log data characteristics
	logger.LogParameters("data_info", map[string]interface{}{
		"train_samples":    len(trainData),
		"test_samples":     len(testData),
		"features":         10,
		"train_file":       trainFile,
		"test_file":        testFile,
		"data_loaded_time": time.Now().Format(time.RFC3339),
	})

	// Initialize model with fixed seed
	const seed = 123
	var linucb *LinUCB
	var epsGreedy *EpsilonGreedy

	if modelType == "linucb" {
		linucb = NewLinUCB(1.0, 10, 5, seed)
		// Log LinUCB parameters
		logger.LogParameters("linucb", map[string]interface{}{
			"alpha":             1.0,
			"context_dimension": 10,
			"num_arms":          5,
			"random_seed":       seed,
			"model_type":        "LinUCB",
			"initialization":    "Identity matrix for A, zero vector for b",
		})
	} else {
		epsGreedy = NewEpsilonGreedy(0.1, 5, seed)
		// Log Epsilon-Greedy parameters
		logger.LogParameters("epsilon_greedy", map[string]interface{}{
			"epsilon":        0.1,
			"num_arms":       5,
			"random_seed":    seed,
			"model_type":     "Epsilon-Greedy",
			"initialization": "Zero counts and values",
		})
	}

	// Initialize history
	history := &History{
		ChosenActions:    make([]int, len(testData)),
		Rewards:          make([]float64, len(testData)),
		Regrets:          make([]float64, len(testData)),
		CumulativeReward: make([]float64, len(testData)),
		CumulativeRegret: make([]float64, len(testData)),
		ContextUsed:      make([][]float64, len(testData)),
	}

	// Train on training data
	for _, record := range trainData {
		context := parseContext(record)
		// bestReward is not used in training, so we remove the declaration
		// bestReward := parseFloatFromRecord(record, 11)

		var arm int
		var reward float64

		if modelType == "linucb" {
			arm = linucb.SelectArm(context)
			rewardCol := 11 + arm // reward_a1 is at index 11
			reward = parseFloatFromRecord(record, rewardCol)
			linucb.Update(arm, context, reward)
		} else {
			arm = epsGreedy.SelectArm()
			rewardCol := 11 + arm
			reward = parseFloatFromRecord(record, rewardCol)
			epsGreedy.Update(arm, reward)
		}
	}

	// Test on test data
	totalReward := 0.0
	totalRegret := 0.0

	for i, record := range testData {
		context := parseContext(record)
		bestReward := parseFloatFromRecord(record, 11)

		var arm int
		var reward float64

		if modelType == "linucb" {
			arm = linucb.SelectArm(context)
		} else {
			arm = epsGreedy.SelectArm()
		}

		rewardCol := 11 + arm
		reward = parseFloatFromRecord(record, rewardCol)
		regret := bestReward - reward

		totalReward += reward
		totalRegret += regret

		history.ChosenActions[i] = arm
		history.Rewards[i] = reward
		history.Regrets[i] = regret
		history.CumulativeReward[i] = totalReward
		history.CumulativeRegret[i] = totalRegret
		history.ContextUsed[i] = context

		// Update model with test data
		if modelType == "linucb" {
			linucb.Update(arm, context, reward)
		} else {
			epsGreedy.Update(arm, reward)
		}
	}

	return history
}

// Create visualizations
func createVisualizations(linucbHistory, epsilonHistory *History, timestamp string) {
	// Cumulative Reward Plot
	p1 := plot.New()
	p1.Title.Text = "Go Implementation - Cumulative Reward Over Time"
	p1.X.Label.Text = "Step"
	p1.Y.Label.Text = "Cumulative Reward"

	// LinUCB data
	linucbRewardPoints := make(plotter.XYs, len(linucbHistory.CumulativeReward))
	for i, val := range linucbHistory.CumulativeReward {
		linucbRewardPoints[i] = plotter.XY{X: float64(i), Y: val}
	}

	linucbLine, err := plotter.NewLine(linucbRewardPoints)
	if err != nil {
		log.Fatal(err)
	}
	linucbLine.Color = plotutil.Color(0)

	// Epsilon-Greedy data
	epsilonRewardPoints := make(plotter.XYs, len(epsilonHistory.CumulativeReward))
	for i, val := range epsilonHistory.CumulativeReward {
		epsilonRewardPoints[i] = plotter.XY{X: float64(i), Y: val}
	}

	epsilonLine, err := plotter.NewLine(epsilonRewardPoints)
	if err != nil {
		log.Fatal(err)
	}
	epsilonLine.Color = plotutil.Color(1)

	p1.Add(linucbLine, epsilonLine)
	p1.Legend.Add("LinUCB", linucbLine)
	p1.Legend.Add("Epsilon-Greedy", epsilonLine)

	if err := p1.Save(6*vg.Inch, 4*vg.Inch, "go_cumulative_reward_"+timestamp+".png"); err != nil {
		log.Fatal(err)
	}

	// Cumulative Regret Plot
	p2 := plot.New()
	p2.Title.Text = "Go Implementation - Cumulative Regret Over Time"
	p2.X.Label.Text = "Step"
	p2.Y.Label.Text = "Cumulative Regret"

	// LinUCB regret data
	linucbRegretPoints := make(plotter.XYs, len(linucbHistory.CumulativeRegret))
	for i, val := range linucbHistory.CumulativeRegret {
		linucbRegretPoints[i] = plotter.XY{X: float64(i), Y: val}
	}

	linucbRegretLine, err := plotter.NewLine(linucbRegretPoints)
	if err != nil {
		log.Fatal(err)
	}
	linucbRegretLine.Color = plotutil.Color(0)

	// Epsilon-Greedy regret data
	epsilonRegretPoints := make(plotter.XYs, len(epsilonHistory.CumulativeRegret))
	for i, val := range epsilonHistory.CumulativeRegret {
		epsilonRegretPoints[i] = plotter.XY{X: float64(i), Y: val}
	}

	epsilonRegretLine, err := plotter.NewLine(epsilonRegretPoints)
	if err != nil {
		log.Fatal(err)
	}
	epsilonRegretLine.Color = plotutil.Color(1)

	p2.Add(linucbRegretLine, epsilonRegretLine)
	p2.Legend.Add("LinUCB", linucbRegretLine)
	p2.Legend.Add("Epsilon-Greedy", epsilonRegretLine)

	if err := p2.Save(6*vg.Inch, 4*vg.Inch, "go_cumulative_regret_"+timestamp+".png"); err != nil {
		log.Fatal(err)
	}
}

func main() {
	fmt.Println("Starting Go Adaptive Decision Analysis...")
	fmt.Println("Random seed set to: 123")

	// Initialize parameter logger
	logger := NewParameterLogger()
	timestamp := time.Now().Format("20060102_150405")

	// Measure LinUCB performance
	linucbMonitor := NewPerformanceMonitor()
	linucbHistory := ada_model_go("contextual_bandit_train.csv", "contextual_bandit_test.csv", "linucb", logger)
	linucbTime, linucbMemory := linucbMonitor.Stop()

	// Log LinUCB performance
	logger.LogPerformance("linucb_model", map[string]interface{}{
		"runtime_seconds":   linucbTime.Seconds(),
		"memory_used_bytes": linucbMemory,
		"memory_used_mb":    float64(linucbMemory) / 1024 / 1024,
		"timestamp":         time.Now().Format(time.RFC3339),
	})

	// Log LinUCB results
	logger.LogPerformance("linucb_results", map[string]interface{}{
		"final_cumulative_reward": linucbHistory.CumulativeReward[len(linucbHistory.CumulativeReward)-1],
		"final_cumulative_regret": linucbHistory.CumulativeRegret[len(linucbHistory.CumulativeRegret)-1],
		"average_reward":          average(linucbHistory.Rewards),
		"average_regret":          average(linucbHistory.Regrets),
		"total_decisions":         len(linucbHistory.ChosenActions),
	})

	// Measure Epsilon-Greedy performance
	epsilonMonitor := NewPerformanceMonitor()
	epsilonHistory := ada_model_go("contextual_bandit_train.csv", "contextual_bandit_test.csv", "epsilon", logger)
	epsilonTime, epsilonMemory := epsilonMonitor.Stop()

	// Log Epsilon-Greedy performance
	logger.LogPerformance("epsilon_greedy_model", map[string]interface{}{
		"runtime_seconds":   epsilonTime.Seconds(),
		"memory_used_bytes": epsilonMemory,
		"memory_used_mb":    float64(epsilonMemory) / 1024 / 1024,
		"timestamp":         time.Now().Format(time.RFC3339),
	})

	// Log Epsilon-Greedy results
	logger.LogPerformance("epsilon_greedy_results", map[string]interface{}{
		"final_cumulative_reward": epsilonHistory.CumulativeReward[len(epsilonHistory.CumulativeReward)-1],
		"final_cumulative_regret": epsilonHistory.CumulativeRegret[len(epsilonHistory.CumulativeRegret)-1],
		"average_reward":          average(epsilonHistory.Rewards),
		"average_regret":          average(epsilonHistory.Regrets),
		"total_decisions":         len(epsilonHistory.ChosenActions),
	})

	// Create visualizations
	createVisualizations(linucbHistory, epsilonHistory, timestamp)

	// Export parameters and performance metrics
	if err := logger.ExportToExcel("go_analysis_parameters_" + timestamp + ".xlsx"); err != nil {
		log.Fatal("Error exporting to Excel:", err)
	}

	// Print summary
	fmt.Println("\n=== GO ANALYSIS COMPLETED ===")
	fmt.Printf("LinUCB - Final Reward: %.4f\n", linucbHistory.CumulativeReward[len(linucbHistory.CumulativeReward)-1])
	fmt.Printf("Epsilon-Greedy - Final Reward: %.4f\n", epsilonHistory.CumulativeReward[len(epsilonHistory.CumulativeReward)-1])
	fmt.Printf("LinUCB Runtime: %v\n", linucbTime)
	fmt.Printf("Epsilon-Greedy Runtime: %v\n", epsilonTime)
	fmt.Printf("Parameters exported to: go_analysis_parameters_%s.xlsx\n", timestamp)
}

// Helper function to calculate average
func average(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}
