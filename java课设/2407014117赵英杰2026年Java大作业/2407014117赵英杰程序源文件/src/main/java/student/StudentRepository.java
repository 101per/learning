package student;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class StudentRepository {
    private final Path dataPath;

    public StudentRepository(Path dataPath) {
        this.dataPath = dataPath;
    }

    public List<Student> load() throws IOException {
        List<Student> students = new ArrayList<>();
        if (!Files.exists(dataPath)) {
            return students;
        }

        try (BufferedReader reader = Files.newBufferedReader(dataPath, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    students.add(parseStudent(line));
                }
            }
        }
        return students;
    }

    public void save(List<Student> students) throws IOException {
        Path parent = dataPath.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(dataPath, StandardCharsets.UTF_8)) {
            for (Student student : students) {
                writer.write(toLine(student));
                writer.newLine();
            }
        }
    }

    static String toLine(Student student) {
        return String.join("\t",
                student.getId(),
                student.getName(),
                Double.toString(student.getEnglish()),
                Double.toString(student.getAdvancedMath()),
                Double.toString(student.getCLanguage()),
                Double.toString(student.getJavaProgramming()));
    }

    private Student parseStudent(String line) {
        String[] parts = line.split("\t", -1);
        if (parts.length != 6) {
            throw new IllegalArgumentException("数据文件格式错误: " + line);
        }
        return new Student(
                parts[0],
                parts[1],
                Double.parseDouble(parts[2]),
                Double.parseDouble(parts[3]),
                Double.parseDouble(parts[4]),
                Double.parseDouble(parts[5]));
    }
}
