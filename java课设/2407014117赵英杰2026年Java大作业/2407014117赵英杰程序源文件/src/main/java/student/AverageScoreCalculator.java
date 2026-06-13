package student;

public class AverageScoreCalculator implements ScoreCalculator {
    @Override
    public String getName() {
        return "平均成绩";
    }

    @Override
    public double calculate(Student student) {
        return student.getAverageScore();
    }
}
