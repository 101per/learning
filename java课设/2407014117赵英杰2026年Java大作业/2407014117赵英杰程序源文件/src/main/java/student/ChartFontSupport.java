package student;

import java.awt.Font;
import java.awt.GraphicsEnvironment;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PiePlot;
import org.jfree.chart.plot.Plot;

public final class ChartFontSupport {
    private static final String[] CHINESE_FONT_CANDIDATES = {
            "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Dialog"
    };
    private static final Font CHART_FONT = new Font(resolveFontFamily(), Font.PLAIN, 14);
    private static final Font TITLE_FONT = CHART_FONT.deriveFont(Font.BOLD, 18f);

    private ChartFontSupport() {
    }

    public static Font chartFont() {
        return CHART_FONT;
    }

    public static void applyChineseFont(JFreeChart chart) {
        if (chart.getTitle() != null) {
            chart.getTitle().setFont(TITLE_FONT);
        }
        if (chart.getLegend() != null) {
            chart.getLegend().setItemFont(CHART_FONT);
        }
        Plot plot = chart.getPlot();
        if (plot instanceof CategoryPlot categoryPlot) {
            applyCategoryPlotFont(categoryPlot);
        } else if (plot instanceof PiePlot<?> piePlot) {
            piePlot.setLabelFont(CHART_FONT);
            piePlot.setNoDataMessageFont(CHART_FONT);
        }
    }

    private static void applyCategoryPlotFont(CategoryPlot plot) {
        CategoryAxis domainAxis = plot.getDomainAxis();
        if (domainAxis != null) {
            domainAxis.setLabelFont(CHART_FONT);
            domainAxis.setTickLabelFont(CHART_FONT);
        }
        ValueAxis rangeAxis = plot.getRangeAxis();
        if (rangeAxis != null) {
            rangeAxis.setLabelFont(CHART_FONT);
            rangeAxis.setTickLabelFont(CHART_FONT);
        }
    }

    private static String resolveFontFamily() {
        Set<String> availableFonts = Stream.of(
                        GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames())
                .collect(Collectors.toSet());
        for (String font : CHINESE_FONT_CANDIDATES) {
            if (availableFonts.contains(font)) {
                return font;
            }
        }
        return Font.SANS_SERIF;
    }
}
