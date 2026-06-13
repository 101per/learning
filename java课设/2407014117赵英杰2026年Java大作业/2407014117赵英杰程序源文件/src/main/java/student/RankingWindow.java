package student;

import java.awt.BorderLayout;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.SwingUtilities;

public class RankingWindow extends JFrame implements StudentChangeListener {
    private final StudentManager manager;
    private final RankingTableModel tableModel = new RankingTableModel();

    public RankingWindow(StudentManager manager) {
        super("实时排名");
        this.manager = manager;
        setSize(760, 420);
        setLocationByPlatform(true);
        setLayout(new BorderLayout());
        add(new JScrollPane(new JTable(tableModel)), BorderLayout.CENTER);
        manager.addChangeListener(this);
        refresh();
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent event) {
                manager.removeChangeListener(RankingWindow.this);
            }
        });
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
    }

    @Override
    public void studentsChanged() {
        SwingUtilities.invokeLater(this::refresh);
    }

    private void refresh() {
        tableModel.setStudents(manager.getRankedStudents());
    }
}
