package student;

import java.util.Objects;

public class Student {
    public static final String[] COURSE_NAMES = {"英语", "高数", "C语言", "Java程序设计"};

    private final String id;
    private final String name;
    private final double english;
    private final double advancedMath;
    private final double cLanguage;
    private final double javaProgramming;

    public Student(String id, String name, double english, double advancedMath,
            double cLanguage, double javaProgramming) {
        this.id = requireText(id, "学号");
        this.name = requireText(name, "姓名");
        this.english = validateScore(english, "英语");
        this.advancedMath = validateScore(advancedMath, "高数");
        this.cLanguage = validateScore(cLanguage, "C语言");
        this.javaProgramming = validateScore(javaProgramming, "Java程序设计");
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public double getEnglish() {
        return english;
    }

    public double getAdvancedMath() {
        return advancedMath;
    }

    public double getCLanguage() {
        return cLanguage;
    }

    public double getJavaProgramming() {
        return javaProgramming;
    }

    public double[] getScores() {
        return new double[] { english, advancedMath, cLanguage, javaProgramming };
    }

    public double getTotalScore() {
        return english + advancedMath + cLanguage + javaProgramming;
    }

    public double getAverageScore() {
        return getTotalScore() / getScores().length;
    }

    public boolean isAlert() {
        for (double score : getScores()) {
            if (score < 60) {
                return true;
            }
        }
        return false;
    }

    public String getAlertCourses() {
        StringBuilder sb = new StringBuilder();
        double[] scores = getScores();
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] < 60) {
                if (sb.length() > 0) {
                    sb.append(",");
                }
                sb.append(COURSE_NAMES[i]);
            }
        }
        return sb.toString();
    }

    private static String requireText(String value, String fieldName) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException(fieldName + "不能为空");
        }
        return value.trim();
    }

    private static double validateScore(double score, String courseName) {
        if (Double.isNaN(score) || score < 0 || score > 100) {
            throw new IllegalArgumentException(courseName + "成绩必须在0到100之间");
        }
        return score;
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        if (!(object instanceof Student student)) {
            return false;
        }
        return Double.compare(student.english, english) == 0
                && Double.compare(student.advancedMath, advancedMath) == 0
                && Double.compare(student.cLanguage, cLanguage) == 0
                && Double.compare(student.javaProgramming, javaProgramming) == 0
                && id.equals(student.id)
                && name.equals(student.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, name, english, advancedMath, cLanguage, javaProgramming);
    }
}
