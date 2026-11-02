/**
 * ============================================================================
 * EduGuard — Student Retention Portal Frontend
 * Module: script.js
 * Description:
 *   Single-page application client built with vanilla JavaScript and Chart.js.
 *   Connects to FastAPI endpoints (e.g. /api/summary), animates KPI numbers,
 *   draws interactive charts, and manages dark/light theme persistence.
 *
 * Author: University AI Project
 * Viva Defense Ready:
 *   - Clean modular functions: state, API client, animation helpers, page renderers.
 *   - No external build tools (Vite/Webpack) required; pure web standards.
 * ============================================================================
 */

// Global Application State
const state = {
    currentPage: "dashboard",
    summaryData: null,
    urgencyWeight: 0.5,
    selectedStudentId: 2237, // default top referral
    theme: localStorage.getItem("eduguard_theme") || "light",
    charts: {
        riskDoughnut: null,
        courseBar: null,
        profileShap: null
    },
    currentProfile: null,
    counselors: {
        data: null,
        filterSpecialty: "all"
    },
    students: {
        riskLevel: "all",
        counselor: "all",
        status: "all",
        search: "",
        sortBy: "priority",
        sortOrder: "desc",
        page: 1,
        pageSize: 25,
        totalRecords: 0,
        totalPages: 1
    },
    modalActiveStudentId: null,
    isLoading: false
};

// ============================================================================
// 1. INITIALIZATION & ROUTING
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initCurrentDate();
    initNavigation();
    initSidebarControls();
    initHeroButtons();
    initButtonRipples();
    initStudentsEventListeners();
    initProfileEventListeners();
    initCounselorsEventListeners();
    initAssignmentsEventListeners();
    initModalEventListeners();
    
    // Fetch and render initial Dashboard view
    loadDashboardData();
});

/**
 * Initializes light/dark theme from localStorage or system preferences.
 */
function initTheme() {
    document.documentElement.setAttribute("data-theme", state.theme);
    const themeBtn = document.getElementById("btn-theme-toggle");
    if (themeBtn) {
        themeBtn.textContent = state.theme === "dark" ? "☀️" : "🌙";
        themeBtn.addEventListener("click", toggleTheme);
    }
}

/**
 * Toggles dark/light theme and updates Chart.js instances with suitable colors.
 */
function toggleTheme() {
    state.theme = state.theme === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", state.theme);
    localStorage.setItem("eduguard_theme", state.theme);
    
    const themeBtn = document.getElementById("btn-theme-toggle");
    if (themeBtn) {
        themeBtn.textContent = state.theme === "dark" ? "☀️" : "🌙";
    }

    showToast(`Switched to ${state.theme} mode`, "info");

    // Re-render charts with updated theme text colors
    if (state.summaryData && state.currentPage === "dashboard") {
        renderCharts(state.summaryData);
    } else if (state.currentProfile && state.currentPage === "profile") {
        renderProfileShapChart(state.currentProfile.local_shap_factors);
    }
}

/**
 * Displays formatted today's calendar date in the sticky navbar.
 */
function initCurrentDate() {
    const datePill = document.getElementById("current-date-pill");
    if (datePill) {
        const options = { month: "short", day: "numeric", year: "numeric" };
        const todayStr = new Date().toLocaleDateString("en-US", options);
        datePill.textContent = `📅 ${todayStr}`;
    }
}

/**
 * Single-Page Navigation router: switches active views without page reloads.
 */
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const targetPage = item.getAttribute("data-page");
            if (targetPage) {
                navigateTo(targetPage);
            }
        });
    });

    // Mobile sidebar toggle
    const toggleBtn = document.getElementById("btn-toggle-sidebar");
    const sidebar = document.getElementById("app-sidebar");
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener("click", () => {
            sidebar.classList.toggle("open");
            sidebar.classList.toggle("collapsed");
        });
    }

    // Brand click returns to dashboard
    const brand = document.getElementById("nav-brand");
    if (brand) {
        brand.addEventListener("click", () => navigateTo("dashboard"));
    }
}

/**
 * Switches current page view and executes the corresponding render function.
 */
function navigateTo(pageName) {
    state.currentPage = pageName;

    // Update active class in sidebar
    document.querySelectorAll(".nav-item").forEach(item => {
        if (item.getAttribute("data-page") === pageName) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });

    // Hide all page containers
    document.querySelectorAll(".page-view").forEach(view => {
        view.style.display = "none";
    });

    // Render target page
    if (pageName === "dashboard") {
        const dashView = document.getElementById("page-dashboard");
        if (dashView) {
            dashView.style.display = "block";
            if (!state.summaryData) {
                loadDashboardData();
            }
        }
    } else if (pageName === "students") {
        const studentsView = document.getElementById("page-students");
        if (studentsView) {
            studentsView.style.display = "block";
            loadStudentsData();
        }
    } else if (pageName === "profile") {
        const profileView = document.getElementById("page-profile");
        if (profileView) {
            profileView.style.display = "block";
            loadStudentProfile(state.selectedStudentId);
        }
    } else if (pageName === "counselors") {
        const counselorsView = document.getElementById("page-counselors");
        if (counselorsView) {
            counselorsView.style.display = "block";
            loadCounselorsData();
        }
    } else if (pageName === "assignments") {
        const assignmentsView = document.getElementById("page-assignments");
        if (assignmentsView) {
            assignmentsView.style.display = "block";
            loadAssignmentsPageData();
        }
    } else {
        renderPlaceholderPage(pageName);
    }

    // Close mobile sidebar if open
    const sidebar = document.getElementById("app-sidebar");
    if (sidebar && window.innerWidth <= 768) {
        sidebar.classList.remove("open");
    }

    // Smooth scroll to app root
    const appRoot = document.getElementById("app-root");
    if (appRoot && window.scrollY > 300) {
        appRoot.scrollIntoView({ behavior: "smooth" });
    }
}

/**
 * Sets up the Urgency Weight slider with debounced API updates.
 */
function initSidebarControls() {
    const slider = document.getElementById("urgency-slider");
    const badge = document.getElementById("urgency-weight-badge");

    if (slider && badge) {
        let debounceTimer;
        slider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            state.urgencyWeight = val;
            badge.textContent = val.toFixed(1);

            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                showToast(`Urgency weight adjusted to ${val.toFixed(1)}`, "info");
                loadDashboardData();
            }, 350);
        });
    }

    // Retry API button in error banner
    const retryBtn = document.getElementById("btn-retry-api");
    if (retryBtn) {
        retryBtn.addEventListener("click", () => {
            loadDashboardData();
        });
    }
}

/**
 * Sets up Landing Hero button interactions.
 */
function initHeroButtons() {
    const btnOpen = document.getElementById("btn-hero-dashboard");
    if (btnOpen) {
        btnOpen.addEventListener("click", () => {
            navigateTo("dashboard");
            const appRoot = document.getElementById("app-root");
            if (appRoot) {
                appRoot.scrollIntoView({ behavior: "smooth" });
            }
        });
    }
}

// ============================================================================
// 2. DATA FETCHING & ERROR HANDLING
// ============================================================================

/**
 * Fetches cohort summary data from the FastAPI backend at /api/summary.
 */
async function loadDashboardData() {
    const errorBanner = document.getElementById("api-error-banner");
    const errorMessage = document.getElementById("api-error-message");
    
    setLoadingState(true);
    if (errorBanner) errorBanner.style.display = "none";

    try {
        const response = await fetch(`/api/summary?urgency_weight=${state.urgencyWeight}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        state.summaryData = data;

        // Render Dashboard components
        renderKPIs(data);
        renderCharts(data);
        renderPriorityTable(data.top_priority || []);
        
        setLoadingState(false);
    } catch (err) {
        console.error("[EduGuard] Failed to fetch summary data:", err);
        setLoadingState(false);
        
        if (errorBanner && errorMessage) {
            errorMessage.innerHTML = `
                Could not connect to <code>http://127.0.0.1:8000/api/summary</code>.<br>
                <b>Details:</b> ${err.message}. Please ensure <code>uvicorn server:app</code> is running.
            `;
            errorBanner.style.display = "flex";
        }
        showToast("Backend connection error. See banner above.", "error");
    }
}

/**
 * Toggles skeleton loader effect on dashboard KPI cards and tables.
 */
function setLoadingState(isLoading) {
    state.isLoading = isLoading;
    const cards = [
        "kpi-total-students",
        "kpi-high-risk",
        "kpi-med-risk",
        "kpi-low-risk",
        "kpi-avg-risk",
        "kpi-unassigned"
    ];

    cards.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            if (isLoading) {
                el.classList.add("skeleton", "skeleton-value");
            } else {
                el.classList.remove("skeleton", "skeleton-value");
            }
        }
    });
}

// ============================================================================
// 3. DASHBOARD PAGE RENDERERS
// ============================================================================

/**
 * Animates number count-up on KPI cards using requestAnimationFrame.
 */
function animateCountUp(elementId, targetValue, duration = 1200, isPercentage = false) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const startTime = performance.now();
    const startValue = 0;

    function step(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Ease-out cubic animation
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = startValue + (targetValue - startValue) * easeOut;

        if (isPercentage) {
            el.textContent = `${current.toFixed(1)}%`;
        } else {
            el.textContent = Math.round(current).toLocaleString();
        }

        if (progress < 1) {
            requestAnimationFrame(step);
        } else {
            if (isPercentage) {
                el.textContent = `${targetValue.toFixed(1)}%`;
            } else {
                el.textContent = targetValue.toLocaleString();
            }
        }
    }

    requestAnimationFrame(step);
}

/**
 * Renders all animated KPI metric cards.
 */
function renderKPIs(data) {
    animateCountUp("kpi-total-students", data.total_students);
    animateCountUp("kpi-high-risk", data.high_risk_count);
    animateCountUp("kpi-med-risk", data.medium_risk_count);
    animateCountUp("kpi-low-risk", data.low_risk_count);
    animateCountUp("kpi-avg-risk", data.average_risk * 100, 1200, true);
    animateCountUp("kpi-unassigned", data.unassigned_count);

    // Also update Hero statistics
    const heroTotal = document.getElementById("hero-total-students");
    const heroHigh = document.getElementById("hero-high-risk");
    if (heroTotal) heroTotal.textContent = data.total_students.toLocaleString();
    if (heroHigh) heroHigh.textContent = data.high_risk_count.toLocaleString();
}

/**
 * Renders both Chart.js visual charts (Doughnut and Bar Chart).
 */
function renderCharts(data) {
    renderRiskDoughnutChart(data.risk_distribution || []);
    renderCourseBarChart(data.risk_by_course || []);
}

/**
 * Renders the Risk Tier Distribution Doughnut Chart.
 */
function renderRiskDoughnutChart(distributionData) {
    const canvas = document.getElementById("chart-risk-distribution");
    if (!canvas) return;

    if (state.charts.riskDoughnut) {
        state.charts.riskDoughnut.destroy();
    }

    const isDark = state.theme === "dark";
    const textColor = isDark ? "#cbd5e1" : "#475569";

    const labels = distributionData.map(d => `${d.tier} Risk`);
    const counts = distributionData.map(d => d.count);
    const colors = ["#ef4444", "#f59e0b", "#10b981"]; // High, Medium, Low

    state.charts.riskDoughnut = new Chart(canvas, {
        type: "doughnut",
        data: {
            labels: labels,
            datasets: [{
                data: counts,
                backgroundColor: colors,
                borderColor: isDark ? "#112238" : "#ffffff",
                borderWidth: 2,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "68%",
            animation: {
                animateScale: true,
                animateRotate: true,
                duration: 1000
            },
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: textColor,
                        font: { family: "Inter", size: 12, weight: 500 },
                        padding: 16,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: isDark ? "#1e3553" : "#0f2b48",
                    padding: 10,
                    titleFont: { family: "Inter", size: 13, weight: "bold" },
                    bodyFont: { family: "Inter", size: 12 },
                    callbacks: {
                        label: function(ctx) {
                            const val = ctx.raw || 0;
                            const total = counts.reduce((a, b) => a + b, 0);
                            const pct = ((val / total) * 100).toFixed(1);
                            return ` ${ctx.label}: ${val.toLocaleString()} students (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Renders the Average Dropout Risk by Degree Course Horizontal Bar Chart.
 */
function renderCourseBarChart(courseData) {
    const canvas = document.getElementById("chart-risk-course");
    if (!canvas) return;

    if (state.charts.courseBar) {
        state.charts.courseBar.destroy();
    }

    const isDark = state.theme === "dark";
    const textColor = isDark ? "#cbd5e1" : "#475569";
    const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)";

    // Top 8 courses with highest average risk
    const topCourses = courseData.slice(0, 8);
    const labels = topCourses.map(c => c.course_name);
    const riskPercentages = topCourses.map(c => (c.average_risk * 100).toFixed(1));

    state.charts.courseBar = new Chart(canvas, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Average Risk (%)",
                data: riskPercentages,
                backgroundColor: isDark ? "rgba(20, 184, 166, 0.85)" : "#0f2b48",
                borderColor: isDark ? "#14b8a6" : "#0f2b48",
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 900, easing: "easeOutQuart" },
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    grid: { color: gridColor },
                    ticks: {
                        color: textColor,
                        font: { family: "Inter", size: 11 },
                        callback: val => `${val}%`
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: textColor,
                        font: { family: "Inter", size: 11, weight: 500 }
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: isDark ? "#1e3553" : "#0f2b48",
                    padding: 10,
                    callbacks: {
                        label: ctx => ` Average Dropout Risk: ${ctx.raw}%`
                    }
                }
            }
        }
    });
}

/**
 * Renders Top 10 Priority Triage Referrals Table with real student values and badges.
 */
function renderPriorityTable(topStudents) {
    const tbody = document.getElementById("table-priority-body");
    if (!tbody) return;

    tbody.innerHTML = "";

    if (!topStudents || topStudents.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 2rem; color: var(--text-muted);">No flagged students available for triage.</td></tr>`;
        return;
    }

    topStudents.forEach((student, index) => {
        const tr = document.createElement("tr");

        // Risk Badge class
        let riskBadgeClass = "badge-low";
        if (student.risk_score >= 0.60) riskBadgeClass = "badge-high";
        else if (student.risk_score >= 0.30) riskBadgeClass = "badge-med";

        // Status Badge class
        let statusBadgeClass = "badge-flagged";
        const stLower = (student.status || "").toLowerCase();
        if (stLower.includes("resolved")) statusBadgeClass = "badge-resolved";
        else if (stLower.includes("contacted")) statusBadgeClass = "badge-contacted";

        tr.innerHTML = `
            <td><b style="color: var(--color-teal);">#${index + 1}</b></td>
            <td><b>Student #${student.student_id}</b></td>
            <td>
                <span class="badge ${riskBadgeClass}">
                    ${(student.risk_score * 100).toFixed(1)}%
                </span>
            </td>
            <td><code>${student.urgency.toFixed(2)}</code></td>
            <td><b style="color: var(--color-navy);">${student.referral_priority.toFixed(4)}</b></td>
            <td>${student.assigned_counselor || "Unassigned"}</td>
            <td><span class="badge ${statusBadgeClass}">● ${student.status || "Flagged"}</span></td>
            <td style="text-align: right;">
                <button class="btn btn-sm btn-view" onclick="viewStudentProfile(${student.student_id})">
                    View Profile →
                </button>
            </td>
        `;

        tbody.appendChild(tr);
    });
}

/**
 * Handler for the "View" button in priority table:
 * Stores selected student and switches view to Student Profile.
 */
function viewStudentProfile(studentId) {
    state.selectedStudentId = studentId;
    showToast(`Opening Profile for Student #${studentId}`, "success");
    navigateTo("profile");
}

// ============================================================================
// 3B. STUDENTS DIRECTORY PAGE LOGIC (Step 3)
// ============================================================================

/**
 * Fetches filtered, sorted, and paginated student records from /api/students.
 */
async function loadStudentsData() {
    const tbody = document.getElementById("students-table-body");
    const counter = document.getElementById("students-counter");
    if (!tbody) return;

    // Show shimmering skeleton loading rows
    tbody.innerHTML = `
        <tr><td colspan="8"><div class="skeleton skeleton-row"></div></td></tr>
        <tr><td colspan="8"><div class="skeleton skeleton-row"></div></td></tr>
        <tr><td colspan="8"><div class="skeleton skeleton-row"></div></td></tr>
    `;

    try {
        const queryParams = new URLSearchParams({
            risk_level: state.students.riskLevel,
            counselor: state.students.counselor,
            status: state.students.status,
            search: state.students.search,
            sort_by: state.students.sortBy,
            sort_order: state.students.sortOrder,
            page: state.students.page,
            page_size: state.students.pageSize,
            urgency_weight: state.urgencyWeight
        });

        const response = await fetch(`/api/students?${queryParams.toString()}`);
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        state.students.totalRecords = data.total;
        state.students.totalPages = data.total_pages;

        if (counter) {
            const start = data.total > 0 ? (data.page - 1) * data.page_size + 1 : 0;
            const end = Math.min(data.page * data.page_size, data.total);
            counter.textContent = `Showing ${start.toLocaleString()} – ${end.toLocaleString()} of ${data.total.toLocaleString()} students`;
        }

        renderStudentsTable(data.students);
        updatePaginationControls(data.page, data.total_pages);

    } catch (err) {
        console.error("[EduGuard] Failed to fetch students list:", err);
        tbody.innerHTML = `
            <tr>
                <td colspan="8" style="text-align: center; padding: 2.5rem; color: var(--color-high);">
                    ⚠️ Failed to load students directory: ${err.message}. Please verify the backend is online.
                </td>
            </tr>
        `;
        showToast("Error loading students data.", "error");
    }
}

/**
 * Renders the student directory table rows with badges and action buttons.
 */
function renderStudentsTable(students) {
    const tbody = document.getElementById("students-table-body");
    if (!tbody) return;

    tbody.innerHTML = "";

    if (!students || students.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" style="text-align: center; padding: 3rem; color: var(--text-muted);">
                    🔍 No student records matched your current filters. Try resetting the search or filter criteria.
                </td>
            </tr>
        `;
        return;
    }

    students.forEach(student => {
        const tr = document.createElement("tr");

        let riskBadgeClass = "badge-low";
        if (student.risk_score >= 0.60) riskBadgeClass = "badge-high";
        else if (student.risk_score >= 0.30) riskBadgeClass = "badge-med";

        let statusBadgeClass = "badge-flagged";
        const stLower = (student.status || "").toLowerCase();
        if (stLower.includes("resolved")) statusBadgeClass = "badge-resolved";
        else if (stLower.includes("contacted")) statusBadgeClass = "badge-contacted";

        const outcomeText = student.true_label === 1 ? "Dropout" : "Retained";
        const outcomeColor = student.true_label === 1 ? "var(--color-high)" : "var(--color-low)";

        tr.innerHTML = `
            <td><b>#${student.student_id}</b></td>
            <td>
                <span class="badge ${riskBadgeClass}">
                    ${(student.risk_score * 100).toFixed(1)}% (${student.risk_tier})
                </span>
            </td>
            <td><code>${student.urgency.toFixed(2)}</code></td>
            <td><b style="color: var(--color-navy);">${student.referral_priority.toFixed(4)}</b></td>
            <td><span>${student.assigned_counselor || "Unassigned"}</span></td>
            <td><span class="badge ${statusBadgeClass}">● ${student.status || "Flagged"}</span></td>
            <td><span style="font-weight: 600; color: ${outcomeColor};">${outcomeText}</span></td>
            <td style="text-align: right; padding-right: 1.5rem; white-space: nowrap;">
                <button class="btn btn-sm btn-view" onclick="viewStudentProfile(${student.student_id})" title="View Profile & SHAP breakdown">
                    View
                </button>
                <button class="btn btn-sm btn-outline" style="background: var(--bg-surface); color: var(--text-secondary); border: 1px solid var(--border-card); margin-left: 6px;" onclick="openAssignModal(${student.student_id}, '${escapeSingleQuotes(student.assigned_counselor)}')" title="Assign Advisor">
                    Assign
                </button>
            </td>
        `;

        tbody.appendChild(tr);
    });
}

/**
 * Escapes single quotes for inline JS attributes.
 */
function escapeSingleQuotes(str) {
    return (str || "").replace(/'/g, "\\'");
}

/**
 * Updates previous/next pagination buttons and page indicator.
 */
function updatePaginationControls(currentPage, totalPages) {
    const btnPrev = document.getElementById("btn-page-prev");
    const btnNext = document.getElementById("btn-page-next");
    const indicator = document.getElementById("page-indicator");

    if (btnPrev) btnPrev.disabled = (currentPage <= 1);
    if (btnNext) btnNext.disabled = (currentPage >= totalPages);
    if (indicator) indicator.textContent = `Page ${currentPage} of ${totalPages || 1}`;
}

/**
 * Sets up event listeners for filters, search, pagination, and CSV export.
 */
function initStudentsEventListeners() {
    // 1. Search with 300ms debounce
    const searchInput = document.getElementById("filter-search");
    if (searchInput) {
        let debounceTimer;
        searchInput.addEventListener("input", (e) => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                state.students.search = e.target.value.trim();
                state.students.page = 1;
                loadStudentsData();
            }, 300);
        });
    }

    // 2. Risk Filter
    const riskSelect = document.getElementById("filter-risk");
    if (riskSelect) {
        riskSelect.addEventListener("change", (e) => {
            state.students.riskLevel = e.target.value;
            state.students.page = 1;
            loadStudentsData();
        });
    }

    // 3. Counselor Filter
    const counselorSelect = document.getElementById("filter-counselor");
    if (counselorSelect) {
        counselorSelect.addEventListener("change", (e) => {
            state.students.counselor = e.target.value;
            state.students.page = 1;
            loadStudentsData();
        });
    }

    // 4. Status Filter
    const statusSelect = document.getElementById("filter-status");
    if (statusSelect) {
        statusSelect.addEventListener("change", (e) => {
            state.students.status = e.target.value;
            state.students.page = 1;
            loadStudentsData();
        });
    }

    // 5. Sort By
    const sortBySelect = document.getElementById("filter-sort-by");
    if (sortBySelect) {
        sortBySelect.addEventListener("change", (e) => {
            state.students.sortBy = e.target.value;
            state.students.page = 1;
            loadStudentsData();
        });
    }

    // 6. Sort Order
    const sortOrderSelect = document.getElementById("filter-sort-order");
    if (sortOrderSelect) {
        sortOrderSelect.addEventListener("change", (e) => {
            state.students.sortOrder = e.target.value;
            state.students.page = 1;
            loadStudentsData();
        });
    }

    // 7. Page Size
    const pageSizeSelect = document.getElementById("students-page-size");
    if (pageSizeSelect) {
        pageSizeSelect.addEventListener("change", (e) => {
            state.students.pageSize = parseInt(e.target.value, 10);
            state.students.page = 1;
            loadStudentsData();
        });
    }

    // 8. Pagination Prev / Next Buttons
    const btnPrev = document.getElementById("btn-page-prev");
    const btnNext = document.getElementById("btn-page-next");
    if (btnPrev) {
        btnPrev.addEventListener("click", () => {
            if (state.students.page > 1) {
                state.students.page--;
                loadStudentsData();
            }
        });
    }
    if (btnNext) {
        btnNext.addEventListener("click", () => {
            if (state.students.page < state.students.totalPages) {
                state.students.page++;
                loadStudentsData();
            }
        });
    }

    // 9. Export CSV Button
    const btnExport = document.getElementById("btn-export-students");
    if (btnExport) {
        btnExport.addEventListener("click", downloadStudentsCSV);
    }
}

/**
 * Downloads the currently filtered student list as a CSV file.
 */
function downloadStudentsCSV() {
    showToast("Generating CSV export from backend...", "info");
    const params = new URLSearchParams({
        risk_level: state.students.riskLevel,
        counselor: state.students.counselor,
        status: state.students.status,
        search: state.students.search,
        urgency_weight: state.urgencyWeight
    });
    window.location.href = `/api/export.csv?${params.toString()}`;
}

// ============================================================================
// 3C. ADVISOR ASSIGNMENT MODAL POPUP
// ============================================================================

/**
 * Opens the Advisor Assignment Modal Dialog for a specific student.
 */
function openAssignModal(studentId, currentCounselor) {
    state.modalActiveStudentId = studentId;

    const modal = document.getElementById("assign-modal");
    const studentName = document.getElementById("modal-student-name");
    const select = document.getElementById("modal-counselor-select");

    if (studentName) studentName.textContent = `Student #${studentId}`;
    if (select) {
        select.value = currentCounselor || "Unassigned";
    }

    if (modal) {
        modal.style.display = "flex";
    }
}

/**
 * Closes the Advisor Assignment Modal Dialog.
 */
function closeAssignModal() {
    const modal = document.getElementById("assign-modal");
    if (modal) {
        modal.style.display = "none";
    }
    state.modalActiveStudentId = null;
}

/**
 * Initializes modal dialog event listeners (backdrop, close button, confirm).
 */
function initModalEventListeners() {
    const modal = document.getElementById("assign-modal");
    const closeBtn = document.getElementById("btn-close-assign-modal");
    const cancelBtn = document.getElementById("btn-cancel-assign");
    const confirmBtn = document.getElementById("btn-confirm-assign");

    if (closeBtn) closeBtn.addEventListener("click", closeAssignModal);
    if (cancelBtn) cancelBtn.addEventListener("click", closeAssignModal);

    // Close on backdrop click outside dialog
    if (modal) {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) closeAssignModal();
        });
    }

    if (confirmBtn) {
        confirmBtn.addEventListener("click", submitAssignStudent);
    }
}

/**
 * Submits the assignment to POST /api/assign/{student_id} and updates the table.
 */
async function submitAssignStudent() {
    const studentId = state.modalActiveStudentId;
    const select = document.getElementById("modal-counselor-select");
    if (!studentId || !select) return;

    const chosenCounselor = select.value;
    const confirmBtn = document.getElementById("btn-confirm-assign");
    if (confirmBtn) confirmBtn.disabled = true;

    try {
        const response = await fetch(`/api/assign/${studentId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ counselor: chosenCounselor })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || `HTTP ${response.status}`);
        }

        closeAssignModal();
        showToast(`Student #${studentId} assigned to ${chosenCounselor}!`, "success");

        // Reload data to reflect change
        loadStudentsData();
        if (state.currentPage === "dashboard") {
            loadDashboardData();
        } else if (state.currentPage === "profile") {
            loadStudentProfile(studentId);
        }

    } catch (err) {
        console.error("[EduGuard] Failed to assign student:", err);
        showToast(`Assignment failed: ${err.message}`, "error");
    } finally {
        if (confirmBtn) confirmBtn.disabled = false;
    }
}

// ============================================================================
// 3D. STUDENT PROFILE PAGE LOGIC (Step 4)
// ============================================================================

/**
 * Initializes event listeners for the Student Profile page:
 * - Back to Directory navigation
 * - Student ID quick switcher (Prev, Next, Go, and Enter key)
 * - Auto-assign specialist based on top risk domain
 * - Advisor reassignment modal trigger
 * - Case status progression (Mark Contacted, Mark Resolved)
 * - Counselor Briefing text file download
 */
function initProfileEventListeners() {
    // 1. Back button returns to Students Directory
    const backBtn = document.getElementById("btn-profile-back");
    if (backBtn) {
        backBtn.addEventListener("click", () => navigateTo("students"));
    }

    // 2. Student ID Switcher Controls (Prev, Next, Go)
    const prevBtn = document.getElementById("btn-prev-student");
    const nextBtn = document.getElementById("btn-next-student");
    const goBtn = document.getElementById("btn-go-student");
    const input = document.getElementById("profile-student-input");

    if (prevBtn) {
        prevBtn.addEventListener("click", () => {
            if (state.selectedStudentId > 0) {
                loadStudentProfile(state.selectedStudentId - 1);
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener("click", () => {
            if (state.selectedStudentId < 4423) {
                loadStudentProfile(state.selectedStudentId + 1);
            }
        });
    }

    if (goBtn && input) {
        const handleGo = () => {
            const raw = String(input.value || "").trim().replace(/^#/, "").replace(/^student\s*#?/i, "");
            const val = parseInt(raw, 10);
            if (!isNaN(val) && val >= 0 && val <= 4423) {
                loadStudentProfile(val);
            } else {
                showToast("Please enter a valid Student ID between 0 and 4423.", "warning");
            }
        };
        goBtn.addEventListener("click", handleGo);
        input.addEventListener("keyup", (e) => {
            if (e.key === "Enter") handleGo();
        });
    }

    // 3. Profile Action Toolbar
    const autoBtn = document.getElementById("btn-profile-auto-assign");
    if (autoBtn) {
        autoBtn.addEventListener("click", handleProfileAutoAssign);
    }

    const reassignBtn = document.getElementById("btn-profile-reassign");
    if (reassignBtn) {
        reassignBtn.addEventListener("click", () => {
            const currentCounselor = state.currentProfile ? state.currentProfile.assigned_counselor : "Unassigned";
            openAssignModal(state.selectedStudentId, currentCounselor);
        });
    }

    const contactedBtn = document.getElementById("btn-profile-mark-contacted");
    if (contactedBtn) {
        contactedBtn.addEventListener("click", () => {
            updateStudentStatus(state.selectedStudentId, "Contacted");
        });
    }

    const resolvedBtn = document.getElementById("btn-profile-mark-resolved");
    if (resolvedBtn) {
        resolvedBtn.addEventListener("click", () => {
            updateStudentStatus(state.selectedStudentId, "Resolved");
        });
    }

    // 4. Download Briefing
    const downloadBtn = document.getElementById("btn-download-briefing");
    if (downloadBtn) {
        downloadBtn.addEventListener("click", downloadStudentBriefing);
    }
}

/**
 * Loads comprehensive profile for an individual student from /api/students/{id}.
 */
async function loadStudentProfile(studentId) {
    state.selectedStudentId = studentId;

    const input = document.getElementById("profile-student-input");
    if (input) input.value = studentId;

    const pre = document.getElementById("profile-briefing-text");
    if (pre) pre.textContent = `Loading SHAP attributions and briefing for Student #${studentId}...`;

    try {
        const response = await fetch(`/api/students/${studentId}?urgency_weight=${state.urgencyWeight}`);
        if (!response.ok) {
            throw new Error(`Student #${studentId} not found (HTTP ${response.status})`);
        }

        const data = await response.json();
        state.currentProfile = data;

        renderProfileDetails(data);

    } catch (err) {
        console.error("[EduGuard] Failed to load student profile:", err);
        showToast(err.message, "error");
        if (pre) pre.textContent = `⚠️ Error loading profile: ${err.message}`;
    }
}

/**
 * Renders all sections of the Student Profile page.
 */
function renderProfileDetails(data) {
    // 1. Identity & Avatar
    const avatar = document.getElementById("profile-avatar");
    const name = document.getElementById("profile-student-id-display");
    const riskBadge = document.getElementById("profile-risk-badge");
    const statusBadge = document.getElementById("profile-status-badge");
    const outcomePill = document.getElementById("profile-outcome-pill");

    if (avatar) avatar.textContent = `#${data.student_id}`;
    if (name) name.textContent = `Student #${data.student_id}`;

    if (riskBadge) {
        let badgeClass = "badge-low";
        if (data.risk_score >= 0.60) badgeClass = "badge-high";
        else if (data.risk_score >= 0.30) badgeClass = "badge-med";
        riskBadge.className = `badge ${badgeClass}`;
        riskBadge.textContent = `${(data.risk_score * 100).toFixed(1)}% ${data.risk_tier} Risk`;
    }

    if (statusBadge) {
        let sClass = "badge-flagged";
        const stLower = (data.status || "").toLowerCase();
        if (stLower.includes("resolved")) sClass = "badge-resolved";
        else if (stLower.includes("contacted")) sClass = "badge-contacted";
        statusBadge.className = `badge ${sClass}`;
        statusBadge.textContent = `● ${data.status}`;
    }

    if (outcomePill) {
        const isDropout = data.true_label === 1;
        outcomePill.textContent = `Benchmark: ${isDropout ? 'Dropout' : 'Retained'}`;
        outcomePill.style.color = isDropout ? 'var(--color-high)' : 'var(--color-low)';
        outcomePill.style.borderColor = isDropout ? 'var(--color-high-border)' : 'var(--color-low-border)';
    }

    // 2. Header KPI Stats Grid
    const kpiRisk = document.getElementById("profile-kpi-risk");
    const kpiTier = document.getElementById("profile-kpi-tier");
    const kpiUrg = document.getElementById("profile-kpi-urgency");
    const kpiPri = document.getElementById("profile-kpi-priority");
    const kpiAdv = document.getElementById("profile-kpi-advisor");
    const kpiAdvSub = document.getElementById("profile-kpi-advisor-sub");

    if (kpiRisk) kpiRisk.textContent = `${(data.risk_score * 100).toFixed(1)}%`;
    if (kpiTier) kpiTier.textContent = `${data.risk_tier} Risk Tier`;
    if (kpiUrg) kpiUrg.textContent = data.urgency.toFixed(2);
    if (kpiPri) kpiPri.textContent = data.referral_priority.toFixed(4);
    if (kpiAdv) kpiAdv.textContent = data.assigned_counselor || "Unassigned";
    if (kpiAdvSub) {
        kpiAdvSub.textContent = (data.assigned_counselor && data.assigned_counselor !== "Unassigned")
            ? "Specialist Assigned"
            : "Awaiting outreach";
    }

    // 3. Render Local SHAP Horizontal Bar Chart
    renderProfileShapChart(data.local_shap_factors);

    // 4. Render Top 3 Positive Drivers Detail Cards
    renderTopFactorsList(data.top_risk_factors);

    // 5. Render Counselor Briefing Box & Enforce Guardrail
    const warningBox = document.getElementById("profile-briefing-warning");
    const briefingText = document.getElementById("profile-briefing-text");
    const briefingBox = document.getElementById("profile-briefing-box");

    if (!data.has_risk_factors) {
        // Enforce NoRiskFactorsError Guardrail
        if (warningBox) {
            warningBox.style.display = "flex";
            warningBox.innerHTML = `
                <span style="font-size: 1.25rem;">⚠️</span>
                <div>
                    <strong>Guardrail Enforced:</strong> ${data.warning || "No risk-increasing factors exist for this student (all SHAP values ≤ 0). No generic briefing will be generated."}
                </div>
            `;
        }
        if (briefingText) {
            briefingText.textContent = `[No generic briefing generated — Student profile exhibits zero positive risk drivers.]`;
        }
    } else {
        if (warningBox) warningBox.style.display = "none";
        if (briefingText) briefingText.textContent = data.briefing || "No briefing text returned.";
    }

    // 6. Render Recommended Intervention Steps
    renderSuggestedActions(data.suggested_actions);
}

/**
 * Draws the local SHAP feature attribution horizontal bar chart with Chart.js.
 * Highlights risk-increasing features in red/rose and protective factors in teal.
 */
function renderProfileShapChart(localFactors) {
    const canvas = document.getElementById("chart-profile-shap");
    if (!canvas || !localFactors) return;

    if (state.charts.profileShap) {
        state.charts.profileShap.destroy();
    }

    // Sort factors by SHAP value descending
    const sorted = [...localFactors].sort((a, b) => b.shap_value - a.shap_value);

    const isDark = state.theme === "dark";
    const textColor = isDark ? "#cbd5e1" : "#475569";
    const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)";

    const labels = sorted.map(f => {
        const truncated = f.feature.length > 28 ? f.feature.substring(0, 26) + "…" : f.feature;
        return `${truncated} (${f.real_value})`;
    });

    const values = sorted.map(f => f.shap_value);
    const bgColors = sorted.map(f => f.shap_value > 0 ? "rgba(239, 68, 68, 0.85)" : "rgba(13, 148, 136, 0.85)");
    const borderColors = sorted.map(f => f.shap_value > 0 ? "#ef4444" : "#0d9488");

    state.charts.profileShap = new Chart(canvas, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: gridColor },
                    ticks: { color: textColor },
                    title: {
                        display: true,
                        text: "SHAP Impact on Dropout Probability (Δ)",
                        color: textColor,
                        font: { size: 11, weight: 600 }
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: textColor,
                        font: { size: 11 }
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: isDark ? "#1e3553" : "#0f2b48",
                    padding: 12,
                    callbacks: {
                        title: ctx => sorted[ctx[0].dataIndex].feature,
                        label: ctx => {
                            const item = sorted[ctx.dataIndex];
                            const direction = item.shap_value > 0 ? "Increases Risk" : "Protective / Mitigating";
                            return [
                                ` SHAP Impact: ${item.shap_value > 0 ? '+' : ''}${item.shap_value.toFixed(4)} (${direction})`,
                                ` Actual Student Value: ${item.real_value}`
                            ];
                        }
                    }
                }
            }
        }
    });
}

/**
 * Populates the Top 3 Positive Risk Drivers detail list.
 */
function renderTopFactorsList(topFactors) {
    const container = document.getElementById("profile-top-factors-list");
    if (!container) return;

    if (!topFactors || topFactors.length === 0) {
        container.innerHTML = `
            <div style="padding: 1.25rem; text-align: center; color: var(--text-muted); font-size: 0.88rem;">
                No risk-increasing factors detected for this student profile.
            </div>
        `;
        return;
    }

    container.innerHTML = topFactors.map((factor, idx) => {
        const specialtyClass = `specialty-${(factor.specialty || "academic").toLowerCase()}`;
        return `
            <div class="factor-card">
                <div class="factor-info">
                    <div class="factor-name">${idx + 1}. ${factor.feature}</div>
                    <div class="factor-meta">
                        <span>Actual Record: <b class="factor-val-badge">${factor.real_value}</b></span>
                        <span class="factor-specialty-badge ${specialtyClass}">${factor.specialty || "Academic"} Domain</span>
                    </div>
                </div>
                <div class="factor-shap-pill">+${factor.shap_value.toFixed(4)} Δ</div>
            </div>
        `;
    }).join("");
}

/**
 * Populates recommended intervention actions tailored to the student's risk profile.
 */
function renderSuggestedActions(actions) {
    const container = document.getElementById("profile-actions-list");
    if (!container) return;

    if (!actions || actions.length === 0) {
        container.innerHTML = `
            <div style="padding: 1.25rem; text-align: center; color: var(--text-muted); font-size: 0.88rem;">
                No specific intervention steps flagged. Routine academic check-in recommended.
            </div>
        `;
        return;
    }

    container.innerHTML = actions.map(act => {
        let icon = "💡";
        let title = "Advising Support";
        let desc = act;

        if (act.startsWith("Academic:")) {
            icon = "📚";
            title = "Academic Tutoring & Recovery Review";
            desc = act.replace("Academic:", "").trim();
        } else if (act.startsWith("Financial:")) {
            icon = "💳";
            title = "Emergency Financial Aid & Payment Relief";
            desc = act.replace("Financial:", "").trim();
        } else if (act.startsWith("Advising:")) {
            icon = "🤝";
            title = "Retention Counseling & Schedule Flexibility";
            desc = act.replace("Advising:", "").trim();
        }

        return `
            <div class="action-item-card">
                <span class="action-item-icon">${icon}</span>
                <div>
                    <div class="action-item-title">${title}</div>
                    <div class="action-item-desc">${desc}</div>
                </div>
            </div>
        `;
    }).join("");
}

/**
 * Auto-assigns the student to the optimal specialized advisor based on
 * the student's primary positive SHAP risk driver and live counselor caseload capacity.
 */
async function handleProfileAutoAssign() {
    const studentId = state.selectedStudentId;
    if (!state.currentProfile) return;

    const btn = document.getElementById("btn-profile-auto-assign");
    if (btn) btn.disabled = true;

    try {
        // 1. Fetch live counselor roster
        const rosterRes = await fetch("/api/counselors");
        if (!rosterRes.ok) throw new Error("Could not fetch counselor roster from backend.");
        const rosterData = await rosterRes.json();
        const counselors = rosterData.counselors;

        // 2. Identify target domain from primary positive risk factor
        const topFactor = state.currentProfile.top_risk_factors?.[0];
        const targetSpecialty = topFactor ? topFactor.specialty : "Academic";

        // 3. Find matching specialist with free capacity and lowest current load
        const matchingWithCapacity = counselors.filter(
            c => c.specialty === targetSpecialty && c.current_load < c.capacity
        );

        let chosenCounselor = null;
        if (matchingWithCapacity.length > 0) {
            matchingWithCapacity.sort((a, b) => a.current_load - b.current_load);
            chosenCounselor = matchingWithCapacity[0].name;
        } else {
            // Fallback: any counselor with available capacity
            const anyWithCapacity = counselors.filter(c => c.current_load < c.capacity);
            if (anyWithCapacity.length > 0) {
                anyWithCapacity.sort((a, b) => a.current_load - b.current_load);
                chosenCounselor = anyWithCapacity[0].name;
            } else {
                chosenCounselor = "Unassigned - capacity full";
            }
        }

        // 4. Submit assignment to POST /api/assign/{id}
        const assignRes = await fetch(`/api/assign/${studentId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ counselor: chosenCounselor })
        });

        if (!assignRes.ok) {
            const errData = await assignRes.json();
            throw new Error(errData.detail || "Assignment update rejected.");
        }

        showToast(`Auto-assigned Student #${studentId} to ${chosenCounselor}!`, "success");
        loadStudentProfile(studentId);

    } catch (err) {
        console.error("[EduGuard] Auto-assignment failed:", err);
        showToast(`Auto-assignment failed: ${err.message}`, "error");
    } finally {
        if (btn) btn.disabled = false;
    }
}

/**
 * Updates outreach case status (Contacted, Resolved) via POST /api/status/{id}.
 */
async function updateStudentStatus(studentId, newStatus) {
    try {
        const response = await fetch(`/api/status/${studentId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ status: newStatus })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || `HTTP ${response.status}`);
        }

        showToast(`Student #${studentId} case status marked as "${newStatus}"!`, "success");
        loadStudentProfile(studentId);

    } catch (err) {
        console.error("[EduGuard] Failed to update case status:", err);
        showToast(`Failed to update status: ${err.message}`, "error");
    }
}

/**
 * Downloads the student's official briefing as a .txt file.
 */
function downloadStudentBriefing() {
    if (!state.currentProfile || !state.currentProfile.briefing) {
        showToast("No briefing available to download for this student.", "warning");
        return;
    }

    const text = state.currentProfile.briefing;
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `eduguard_briefing_student_${state.selectedStudentId}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showToast(`Downloaded briefing for Student #${state.selectedStudentId}`, "success");
}

// ============================================================================
// 3E. COUNSELORS ROSTER PAGE LOGIC (Step 5)
// ============================================================================

/**
 * Initializes event listeners for the Counselors page:
 * - Domain specialty filter tabs (All, Academic, Financial, Engagement).
 */
function initCounselorsEventListeners() {
    const tabs = document.querySelectorAll("#counselor-tabs .counselor-tab-btn");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            state.counselors.filterSpecialty = tab.getAttribute("data-specialty") || "all";
            if (state.counselors.data) {
                renderCounselorsGrid(state.counselors.data);
            }
        });
    });
}

/**
 * Loads simulated counselor roster and live caseload metrics from GET /api/counselors.
 */
async function loadCounselorsData() {
    const container = document.getElementById("counselors-cards-container");
    if (!container) return;

    // Show skeleton loaders
    container.innerHTML = `
        <div class="card"><div class="skeleton skeleton-row" style="height: 180px;"></div></div>
        <div class="card"><div class="skeleton skeleton-row" style="height: 180px;"></div></div>
        <div class="card"><div class="skeleton skeleton-row" style="height: 180px;"></div></div>
    `;

    try {
        const response = await fetch("/api/counselors");
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        state.counselors.data = data.counselors;

        // 1. Update Institutional Caseload Overview KPIs
        const totalAdvisors = document.getElementById("counselors-total-count");
        const totalCapacity = document.getElementById("counselors-total-capacity");
        const totalAssigned = document.getElementById("counselors-total-assigned");
        const totalAvailable = document.getElementById("counselors-total-available");
        const totalUtilization = document.getElementById("counselors-total-utilization");

        const assigned = data.total_assigned || 0;
        const capacity = data.total_capacity || 1700;
        const available = Math.max(0, capacity - assigned);
        const utilPct = capacity > 0 ? ((assigned / capacity) * 100).toFixed(1) : "0.0";

        if (totalAdvisors) totalAdvisors.textContent = data.total_counselors;
        if (totalCapacity) totalCapacity.textContent = capacity.toLocaleString();
        if (totalAssigned) totalAssigned.textContent = assigned.toLocaleString();
        if (totalAvailable) totalAvailable.textContent = available.toLocaleString();
        if (totalUtilization) totalUtilization.textContent = `${utilPct}%`;

        // 2. Render Counselor Cards Grid
        renderCounselorsGrid(data.counselors);

    } catch (err) {
        console.error("[EduGuard] Failed to fetch counselors roster:", err);
        container.innerHTML = `
            <div class="card" style="grid-column: 1 / -1; padding: 2rem; text-align: center; color: var(--color-high);">
                ⚠️ Failed to load counselors roster: ${err.message}. Please check that the server is online.
            </div>
        `;
        showToast("Error loading counselors roster.", "error");
    }
}

/**
 * Renders the 6 simulated counselor cards in a responsive grid.
 */
function renderCounselorsGrid(counselors) {
    const container = document.getElementById("counselors-cards-container");
    const countBadge = document.getElementById("counselors-view-count");
    if (!container || !counselors) return;

    // Filter by active specialty tab
    const filter = (state.counselors.filterSpecialty || "all").toLowerCase();
    const filtered = (filter === "all")
        ? counselors
        : counselors.filter(c => (c.specialty || "").toLowerCase() === filter);

    if (countBadge) {
        countBadge.textContent = `Showing ${filtered.length} of ${counselors.length} advisors`;
    }

    if (filtered.length === 0) {
        container.innerHTML = `
            <div class="card" style="grid-column: 1 / -1; padding: 2.5rem; text-align: center; color: var(--text-muted);">
                🔍 No advisors found under the "${filter}" specialty domain.
            </div>
        `;
        return;
    }

    container.innerHTML = filtered.map(c => {
        const spec = (c.specialty || "Academic").toLowerCase();
        let specBadgeClass = "specialty-academic";
        let accentColor = "#0284c7";
        let defaultEmoji = "📚";

        if (spec === "financial") {
            specBadgeClass = "specialty-financial";
            accentColor = "#d97706";
            defaultEmoji = "💳";
        } else if (spec === "engagement") {
            specBadgeClass = "specialty-engagement";
            accentColor = "#9333ea";
            defaultEmoji = "🤝";
        }

        const avatar = c.avatar || defaultEmoji;
        const util = Math.min(100, Math.max(0, c.utilization_pct || 0));

        // Color-coded progress bar threshold
        let progressColor = "#10b981"; // emerald (< 75%)
        if (util >= 90) progressColor = "#ef4444"; // crimson (> 90%)
        else if (util >= 75) progressColor = "#f59e0b"; // amber (75-90%)

        return `
            <div class="counselor-card">
                <div class="counselor-card-accent" style="background: ${accentColor};"></div>
                
                <div>
                    <div class="counselor-card-top">
                        <div class="counselor-identity">
                            <div class="counselor-avatar">${avatar}</div>
                            <div>
                                <h3 class="counselor-name">${c.name}</h3>
                                <div class="counselor-title">${c.title}</div>
                            </div>
                        </div>
                        <span class="factor-specialty-badge ${specBadgeClass}">
                            ${c.specialty} Domain
                        </span>
                    </div>

                    <div class="counselor-contact">
                        <span>✉️</span>
                        <span>${c.email}</span>
                    </div>

                    <!-- Visual Caseload Progress Box -->
                    <div class="counselor-caseload-box">
                        <div class="caseload-row">
                            <span class="caseload-label">Current Caseload</span>
                            <span class="caseload-ratio">${c.current_load} / ${c.capacity} students</span>
                        </div>
                        <div class="caseload-progress-track">
                            <div class="caseload-progress-fill" style="width: ${util.toFixed(1)}%; background: ${progressColor};"></div>
                        </div>
                        <div class="caseload-footer-meta">
                            <span><b>${c.available_slots}</b> slots free</span>
                            <span style="font-weight: 700; color: ${progressColor};">${util.toFixed(1)}% full</span>
                        </div>
                    </div>
                </div>

                <div class="counselor-card-actions">
                    <button class="btn btn-sm btn-outline" style="width: 100%; background: var(--bg-surface); color: var(--color-teal); border: 1px solid var(--border-card);" onclick="viewCounselorCaseload('${escapeSingleQuotes(c.name)}')">
                        👥 View Caseload (${c.current_load}) →
                    </button>
                </div>
            </div>
        `;
    }).join("");
}

/**
 * Deep-links directly to Students Directory filtered to this counselor's caseload.
 */
function viewCounselorCaseload(counselorName) {
    state.students.counselor = counselorName;
    state.students.page = 1;

    // Update the dropdown in the Students Directory page
    const select = document.getElementById("filter-counselor");
    if (select) {
        select.value = counselorName;
    }

    showToast(`Filtering Students Directory by ${counselorName}'s caseload...`, "info");
    navigateTo("students");
}

// ============================================================================
// 3F. ASSIGNMENTS ENGINE PAGE LOGIC (Step 6)
// ============================================================================

/**
 * Initializes event listeners for the Assignments Engine page:
 * - Run Auto-Assignment button
 * - Reset All Assignments button
 * - Export CSV button
 */
function initAssignmentsEventListeners() {
    const runBtn = document.getElementById("btn-run-auto-assign");
    if (runBtn) {
        runBtn.addEventListener("click", executeAutoAssignment);
    }

    const resetBtn = document.getElementById("btn-reset-assignments");
    if (resetBtn) {
        resetBtn.addEventListener("click", executeResetAssignments);
    }

    const exportBtn = document.getElementById("btn-export-assignments");
    if (exportBtn) {
        exportBtn.addEventListener("click", () => {
            showToast("Generating assignments CSV export...", "info");
            window.location.href = `/api/export.csv?urgency_weight=${state.urgencyWeight}`;
        });
    }
}

/**
 * Loads assignment engine metrics, caseload breakdown, and queue preview.
 */
async function loadAssignmentsPageData() {
    try {
        // Fetch counselor caseloads and top summary
        const [counselorsRes, summaryRes] = await Promise.all([
            fetch("/api/counselors"),
            fetch(`/api/summary?urgency_weight=${state.urgencyWeight}`)
        ]);

        if (!counselorsRes.ok || !summaryRes.ok) {
            throw new Error("Failed to load assignment data from API.");
        }

        const cData = await counselorsRes.json();
        const sData = await summaryRes.json();

        // 1. Update KPI Cards
        const totalEligible = sData.eligible_for_advising || 1937;
        const totalCapacity = cData.total_capacity || 1700;
        const totalAssigned = cData.total_assigned || 0;
        const overflow = Math.max(0, totalEligible - totalCapacity);

        const elEligible = document.getElementById("kpi-assign-eligible");
        const elCapacity = document.getElementById("kpi-assign-capacity");
        const elAssigned = document.getElementById("kpi-assign-assigned");
        const elAssignedSub = document.getElementById("kpi-assign-assigned-sub");
        const elOverflow = document.getElementById("kpi-assign-overflow");

        if (elEligible) elEligible.textContent = totalEligible.toLocaleString();
        if (elCapacity) elCapacity.textContent = totalCapacity.toLocaleString();
        if (elAssigned) {
            elAssigned.textContent = totalAssigned.toLocaleString();
            elAssigned.style.color = totalAssigned > 0 ? "var(--color-low)" : "var(--text-primary)";
        }
        if (elAssignedSub) {
            if (totalAssigned >= totalCapacity) elAssignedSub.textContent = "100% Capacity Reached";
            else if (totalAssigned > 0) elAssignedSub.textContent = "Partial Triage Active";
            else elAssignedSub.textContent = "Awaiting Assignment Run";
        }
        if (elOverflow) {
            elOverflow.textContent = totalAssigned >= totalCapacity ? overflow.toLocaleString() : "0";
        }

        // 2. Render Caseload Distribution Summary Table
        renderAssignmentsSummaryTable(cData.counselors);

        // 3. Render High-Priority Triage Queue Preview Table
        renderAssignmentsQueueTable(sData.top_priority);

    } catch (err) {
        console.error("[EduGuard] Failed to load assignments page data:", err);
        showToast("Error loading assignments data.", "error");
    }
}

/**
 * Renders the advisor caseload summary table.
 */
function renderAssignmentsSummaryTable(counselors) {
    const tbody = document.getElementById("table-assignments-summary-body");
    if (!tbody || !counselors) return;

    tbody.innerHTML = counselors.map(c => {
        const spec = (c.specialty || "Academic").toLowerCase();
        let specBadgeClass = "specialty-academic";
        if (spec === "financial") specBadgeClass = "specialty-financial";
        else if (spec === "engagement") specBadgeClass = "specialty-engagement";

        const util = Math.min(100, Math.max(0, c.utilization_pct || 0));
        let progressColor = "#10b981";
        if (util >= 90) progressColor = "#ef4444";
        else if (util >= 75) progressColor = "#f59e0b";

        return `
            <tr>
                <td>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">${c.avatar || "👤"}</span>
                        <div>
                            <b>${c.name}</b>
                            <div style="font-size: 0.75rem; color: var(--text-muted);">${c.title}</div>
                        </div>
                    </div>
                </td>
                <td>
                    <span class="factor-specialty-badge ${specBadgeClass}">
                        ${c.specialty}
                    </span>
                </td>
                <td><b>${c.current_load}</b> / ${c.capacity}</td>
                <td style="min-width: 140px;">
                    <div class="caseload-progress-track" style="margin: 0;">
                        <div class="caseload-progress-fill" style="width: ${util.toFixed(1)}%; background: ${progressColor};"></div>
                    </div>
                </td>
                <td><span>${c.available_slots} slots open</span></td>
                <td>
                    <span class="badge ${util >= 90 ? 'badge-high' : util >= 75 ? 'badge-med' : 'badge-low'}">
                        ${util.toFixed(1)}%
                    </span>
                </td>
                <td style="text-align: right;">
                    <button class="btn btn-sm btn-outline" style="background: var(--bg-surface); color: var(--color-teal); border: 1px solid var(--border-card);" onclick="viewCounselorCaseload('${escapeSingleQuotes(c.name)}')">
                        View Caseload →
                    </button>
                </td>
            </tr>
        `;
    }).join("");
}

/**
 * Renders the High-Priority Triage Queue Preview table (Top 10 referrals).
 */
function renderAssignmentsQueueTable(students) {
    const tbody = document.getElementById("table-assignments-queue-body");
    if (!tbody) return;

    if (!students || students.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 2rem; color: var(--text-muted);">No students in triage queue.</td></tr>`;
        return;
    }

    tbody.innerHTML = students.map((s, idx) => {
        let riskBadgeClass = "badge-low";
        if (s.risk_score >= 0.60) riskBadgeClass = "badge-high";
        else if (s.risk_score >= 0.30) riskBadgeClass = "badge-med";

        let statusBadgeClass = "badge-flagged";
        const stLower = (s.status || "").toLowerCase();
        if (stLower.includes("resolved")) statusBadgeClass = "badge-resolved";
        else if (stLower.includes("contacted")) statusBadgeClass = "badge-contacted";

        return `
            <tr>
                <td><b style="color: var(--color-teal);">#${idx + 1}</b></td>
                <td><b>Student #${s.student_id}</b></td>
                <td><span class="badge ${riskBadgeClass}">${(s.risk_score * 100).toFixed(1)}% (${s.risk_tier})</span></td>
                <td><b style="color: var(--color-navy);">${s.referral_priority.toFixed(4)}</b></td>
                <td>
                    <span style="font-weight: 600;">
                        ${s.assigned_counselor || "Unassigned"}
                    </span>
                </td>
                <td><span class="badge ${statusBadgeClass}">● ${s.status || "Flagged"}</span></td>
                <td style="text-align: right;">
                    <button class="btn btn-sm btn-view" onclick="viewStudentProfile(${s.student_id})">
                        Profile →
                    </button>
                </td>
            </tr>
        `;
    }).join("");
}

/**
 * Executes capacity-constrained auto-assignment for all High and Medium risk students.
 */
async function executeAutoAssignment() {
    const btn = document.getElementById("btn-run-auto-assign");
    const originalText = btn ? btn.innerHTML : "⚡ Run Auto-Assignment";

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = "⏳ Processing 1,937 Students...";
    }

    try {
        const response = await fetch("/api/assign/auto", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                risk_threshold: 0.30,
                urgency_weight: state.urgencyWeight
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || `HTTP ${response.status}`);
        }

        const result = await response.json();
        showToast(`Auto-assignment completed! Processed ${result.processed_students.toLocaleString()} students across 1,700 specialist slots.`, "success");

        // Invalidate cache and reload view
        state.summaryData = null;
        state.counselors.data = null;
        loadAssignmentsPageData();

    } catch (err) {
        console.error("[EduGuard] Auto-assignment failed:", err);
        showToast(`Auto-assignment failed: ${err.message}`, "error");
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    }
}

/**
 * Resets all assignments and case statuses back to unassigned defaults.
 */
async function executeResetAssignments() {
    const confirmed = confirm("Are you sure you want to reset all counselor assignments and student case statuses back to default?");
    if (!confirmed) return;

    const btn = document.getElementById("btn-reset-assignments");
    if (btn) btn.disabled = true;

    try {
        const response = await fetch("/api/reset", {
            method: "POST"
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        showToast("All assignments and statuses have been reset to default.", "info");

        // Invalidate cache and reload view
        state.summaryData = null;
        state.counselors.data = null;
        loadAssignmentsPageData();

    } catch (err) {
        console.error("[EduGuard] Reset failed:", err);
        showToast(`Reset failed: ${err.message}`, "error");
    } finally {
        if (btn) btn.disabled = false;
    }
}

// ============================================================================
// 4. PLACEHOLDER PAGE RENDERER (For Steps 7+)
// ============================================================================

/**
 * Renders informative placeholder view for pages that will be built in subsequent steps.
 */
function renderPlaceholderPage(pageName) {
    const container = document.getElementById(`page-${pageName}`);
    if (!container) return;

    container.style.display = "block";

    const pageMeta = {
        students: {
            title: "Students Directory",
            icon: "👥",
            desc: "Search, filter by risk tier or counselor, and manage all 4,424 student retention records with pagination.",
            step: "Coming in Step 3"
        },
        profile: {
            title: `Student Profile (Selected: #${state.selectedStudentId})`,
            icon: "👤",
            desc: `Detailed game-theoretic SHAP risk factor breakdown, unscaled real values, and official advising briefing for Student #${state.selectedStudentId}.`,
            step: "Coming in Step 4"
        },
        counselors: {
            title: "Advising Team & Caseloads",
            icon: "🧑‍💼",
            desc: "View simulated counselor capacities across Academic, Financial, and Engagement specialties with real-time utilization bars.",
            step: "Coming in Step 5"
        },
        assignments: {
            title: "Capacity-Constrained Assignment Engine",
            icon: "📋",
            desc: "Run automated, priority-ordered auto-assignment matching student SHAP risk factors to specialized advisors.",
            step: "Coming in Step 6"
        },
        reports: {
            title: "Performance & Acceptance Scorecard",
            icon: "📈",
            desc: "Stratified 5-Fold cross-validation results, ACCEPTANCE.md pass/fail verification, and global SHAP insights.",
            step: "Coming in Step 7"
        },
        about: {
            title: "About EduGuard",
            icon: "ℹ️",
            desc: "End-to-end Explainable AI methodology, technological architecture, and viva examination documentation.",
            step: "Project Overview"
        }
    };

    const meta = pageMeta[pageName] || {
        title: pageName,
        icon: "📌",
        desc: "Module details will be connected in subsequent steps.",
        step: "In Progress"
    };

    container.innerHTML = `
        <div class="page-header">
            <h1 class="page-title">${meta.icon} ${meta.title}</h1>
            <p class="page-subtitle">${meta.desc}</p>
        </div>

        <div class="placeholder-view">
            <div class="placeholder-icon">${meta.icon}</div>
            <h2 class="placeholder-title">${meta.title}</h2>
            <p class="placeholder-desc">${meta.desc}</p>
            <div style="display: flex; gap: 10px;">
                <span class="badge badge-med">⚡ ${meta.step}</span>
                <button class="btn btn-sm btn-primary" onclick="navigateTo('dashboard')">
                    ← Back to Dashboard
                </button>
            </div>
        </div>
    `;
}

// ============================================================================
// 5. TOAST NOTIFICATIONS & UI UTILITIES
// ============================================================================

/**
 * Displays an animated floating toast notification.
 */
function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    
    const icons = {
        success: "✅",
        error: "⚠️",
        warning: "🔔",
        info: "ℹ️"
    };

    toast.innerHTML = `
        <span style="font-size: 1.1rem;">${icons[type] || "ℹ️"}</span>
        <span style="flex: 1;">${message}</span>
    `;

    container.appendChild(toast);

    // Auto remove after 3.5 seconds
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(50px)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

/**
 * Adds an interactive wave ripple effect to buttons on click.
 */
function initButtonRipples() {
    document.addEventListener("click", (e) => {
        const btn = e.target.closest(".btn");
        if (!btn) return;

        const circle = document.createElement("span");
        const diameter = Math.max(btn.clientWidth, btn.clientHeight);
        const radius = diameter / 2;

        const rect = btn.getBoundingClientRect();
        circle.style.width = circle.style.height = `${diameter}px`;
        circle.style.left = `${e.clientX - rect.left - radius}px`;
        circle.style.top = `${e.clientY - rect.top - radius}px`;
        circle.classList.add("ripple");

        const ripple = btn.getElementsByClassName("ripple")[0];
        if (ripple) ripple.remove();

        btn.appendChild(circle);
    });
}
