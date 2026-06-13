package student;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Supplier;

public class BackupService implements AutoCloseable {
    private static final DateTimeFormatter FILE_TIME_FORMAT =
            DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss");

    private final Path backupDirectory;
    private final Supplier<List<Student>> studentSupplier;
    private final Consumer<String> statusConsumer;
    private final ScheduledExecutorService executorService;

    public BackupService(Path backupDirectory, Supplier<List<Student>> studentSupplier,
            Consumer<String> statusConsumer) {
        this.backupDirectory = backupDirectory;
        this.studentSupplier = studentSupplier;
        this.statusConsumer = statusConsumer;
        this.executorService = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "student-auto-backup");
            thread.setDaemon(true);
            return thread;
        });
    }

    public void start(long initialDelay, long period, TimeUnit unit) {
        executorService.scheduleAtFixedRate(this::runScheduledBackup, initialDelay, period, unit);
    }

    public Path backupNow() throws IOException {
        Files.createDirectories(backupDirectory);
        Path backupPath = backupDirectory.resolve("students-"
                + LocalDateTime.now().format(FILE_TIME_FORMAT) + ".txt");
        new StudentRepository(backupPath).save(studentSupplier.get());
        return backupPath;
    }

    @Override
    public void close() {
        executorService.shutdownNow();
    }

    private void runScheduledBackup() {
        try {
            Path path = backupNow();
            statusConsumer.accept("最近自动备份: " + path.getFileName());
        } catch (IOException | RuntimeException exception) {
            statusConsumer.accept("自动备份失败: " + exception.getMessage());
        }
    }
}
