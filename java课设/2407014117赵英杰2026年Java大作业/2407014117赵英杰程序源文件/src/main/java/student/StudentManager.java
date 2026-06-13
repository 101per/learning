package student;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CopyOnWriteArrayList;

public class StudentManager {
    private final List<Student> students = new ArrayList<>();
    private final List<StudentChangeListener> listeners = new CopyOnWriteArrayList<>();

    public void setStudents(List<Student> loadedStudents) {
        synchronized (students) {
            students.clear();
            students.addAll(loadedStudents);
        }
        notifyListeners();
    }

    public boolean addStudent(Student student) {
        synchronized (students) {
            if (findByIdInternal(student.getId()).isPresent()) {
                return false;
            }
            students.add(student);
        }
        notifyListeners();
        return true;
    }

    public boolean updateStudent(Student student) {
        synchronized (students) {
            for (int i = 0; i < students.size(); i++) {
                if (students.get(i).getId().equals(student.getId())) {
                    students.set(i, student);
                    notifyListeners();
                    return true;
                }
            }
        }
        return false;
    }

    public boolean deleteStudent(String id) {
        boolean removed;
        synchronized (students) {
            removed = students.removeIf(student -> student.getId().equals(id));
        }
        if (removed) {
            notifyListeners();
        }
        return removed;
    }

    public Optional<Student> findById(String id) {
        synchronized (students) {
            return findByIdInternal(id);
        }
    }

    public List<Student> getAll() {
        synchronized (students) {
            return List.copyOf(students);
        }
    }

    public List<Student> getRankedStudents() {
        return getAll().stream()
                .sorted(Comparator.comparingDouble(Student::getTotalScore).reversed())
                .toList();
    }

    public List<Student> getAlertStudents() {
        return getAll().stream()
                .filter(Student::isAlert)
                .toList();
    }

    public void addChangeListener(StudentChangeListener listener) {
        listeners.add(listener);
    }

    public void removeChangeListener(StudentChangeListener listener) {
        listeners.remove(listener);
    }

    public boolean add(Student student) {
        return addStudent(student);
    }

    public boolean update(Student student) {
        return updateStudent(student);
    }

    public boolean delete(String id) {
        return deleteStudent(id);
    }

    private Optional<Student> findByIdInternal(String id) {
        return students.stream()
                .filter(student -> student.getId().equals(id))
                .findFirst();
    }

    private void notifyListeners() {
        for (StudentChangeListener listener : listeners) {
            listener.studentsChanged();
        }
    }
}
