package student;

public class StandardDeviationCalculator implements ScoreCalculator {
    @Override
    public String getName() {
        return "标准差";
    }

    @Override
    public double calculate(Student student) {
        double[] scores = student.getScores();
        double average = student.getAverageScore();
        double squareSum = 0;
        for (double score : scores) {
            double difference = score - average;
            squareSum += difference * difference;
        }
        return Math.sqrt(squareSum / scores.length);
    }
}
