#include <stdio.h>     // printf
#include <time.h>      // clock_t, clock, CLOCKS_PER_SEC
#include <unistd.h>    // usleep
#include <queue>       // std::queue
#include <thread>      // std::thread

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <opencv2/opencv.hpp>

#include "e2config.hpp"


class Timer {
    public:
        Timer() : beg_(clock_::now()) {}
        void reset() { beg_ = clock_::now(); }
        double elapsed() const { 
            return std::chrono::duration_cast<second_>
                (clock_::now() - beg_).count(); }

    private:
        typedef std::chrono::high_resolution_clock clock_;
        typedef std::chrono::duration<double, std::ratio<1>> second_;
        std::chrono::time_point<clock_> beg_;
};


struct Frame {
    Frame() {}
    Frame(const unsigned int fid, const cv::Mat & src) : frameId(fid) {
        src.copyTo(frame);
    }
    unsigned int frameId;
    cv::Mat frame;
    
};

struct Tag {
    Tag() {}
    Tag(const cv::Mat & src, const unsigned int id, const std::string lb,
        const cv::Rect & r, const cv::Scalar c=cv::Scalar(0, 150, 0)
    ) : uuid(id), label(lb), rect(r), color(c) {
        src(r).copyTo(tmpl);
        whRatio = double(r.width) / double(r.height);
        setSrchRect(src);
    }
    bool setSrchRect(const cv::Mat & src,
        int pt=20, int pr=40, int pb=20, int pl=40) {
        int wsrc = src.cols, hsrc = src.rows;
        bool ret = true;
        int x_, y_, w_, h_, x, y, w, h;

        x_ = rect.x - pr;
        y_ = rect.y - pt;
        w_ = pr + pl + rect.width;
        h_ = pt + pb + rect.height;

        x = std::max(0, x_);
        y = std::max(0, y_);
        w = std::min(w_, wsrc - x);
        h = std::min(h_, hsrc - y);
        
        srchRect = cv::Rect(x, y, w, h);
        if (x_ < 0 || y_ < 0 || w_ > wsrc - x || h_ > hsrc - y) ret = false;

        return ret;
    }

    bool updateTemplate(unsigned int uid, const cv::Mat & src, cv::Rect & rect_) {
        bool ret = true;
        uuid = uid;
        rect = cv::Rect(rect_);
        try {
            src(rect).copyTo(tmpl);
            ret = setSrchRect(src);
        } catch(...) {
            std::cout << "Update template failed!\n";
            ret = false;
        }
        
        return ret;
    }
    unsigned int uuid;
    std::string label;
    cv::Mat tmpl;
    cv::Rect rect;
    double whRatio;
    cv::Rect srchRect;
    cv::Scalar color;
};

static bool keyHandler(const unsigned int delay=1);
static void onMouse(int event, int x, int y, int flag, void* );

void toGaussionBlur(cv::Mat &src, cv::Mat &dst, int ksize);
void toErode(cv::Mat &src, cv::Mat &dst, int ksize);
void toXYGradient(const cv::Mat &src, cv::Mat &dsrc);

double search(const cv::Mat & srchBox, const cv::Mat & tmpl, const double whRatio, cv::Rect & uRect,
    int blurSize=-1, bool searchGrad=false, int erodeSize=-1,
    double lower=0.95, double upper=1.05, double step=0.01,
    cv::Point * where=NULL, double * scale=NULL);

void updateTask(double t=0.03, unsigned int d=1000);
void renderTask(double t=0.03, unsigned int d=1000);
void dumpTask(std::string f, unsigned int t, unsigned int d);


void displayDebugMsg(cv::Mat & src);
void displayLabel(cv::Mat & src);
void displayRect(cv::Mat & src, const cv::Rect & r, cv::Scalar color=cv::Scalar(0, 0, 200), int thickness=2);
void displayTag(cv::Mat & src, const Tag & tag);
void loadConfig(std::string filepath, e2config::Map &cfg);
void loadLabel(std::string filepath);
double parseDouble(std::string key, double defaultVal, e2config::Map &cfg);
int parseInt(std::string key, int defaultVal, e2config::Map &cfg);
bool parseBoolean(std::string key, bool defaultVal, e2config::Map &cfg);


bool precheck(e2config::Map &cfg);

e2config::Map config;

static volatile bool doUpdate = true;
static volatile bool doRender = true;
static volatile bool doDump = true;
static volatile bool showDebug = false;
static volatile bool showLabel = false;
static volatile bool inTagMode = false;
static volatile bool isPlay = false;
static volatile bool inSession = false;

const unsigned int displayTimeout = 20;
unsigned int displayCount = displayTimeout;
unsigned int idxSelectedTag = 0;
unsigned int idxSelectedLabel = 0;
unsigned int inQueMaxSize = 3;
unsigned int renderQueMaxSize = 3;
unsigned int tagQueMaxSize = 10;

static std::vector<Tag> vecTags;
static std::vector<std::string> labels = {"object"};
static cv::Rect rawRect;

static std::queue<Tag> tagQue;
static std::queue<Frame> inQue;
static std::queue<cv::Mat> renderQue;
static std::queue<const Tag> outQue;

int main() {

    loadConfig("config.ini", config);
    std::cout << "Config loaded: \n";
    std::cout << config;

    if (!precheck(config)) {
        std::cout << "Abort. \n";
        return -1;
    }

    std::string filename;
    std::size_t found = config["InputVideo"].rfind("/");
    if (found == std::string::npos) {
        filename = config["InputVideo"] + ".tag";
    } else {
        filename = config["InputVideo"].substr(found + 1) + ".tag";
    }
    std::string outDir = config["OutputDir"];
    if (outDir[outDir.length() - 1] == '/') {
        filename = outDir + filename;
    } else {
        filename = outDir + "/" + filename;
    }

    std::ofstream out(filename);
    out << "";
    out.close();

    double fps = parseDouble("VideoFps", 30.0, config);
    double mainThreadInterval = 1.0 / fps;
    double renderTimeout = parseDouble("RenderTimeout", 0.03, config);
    double updateTimeout = parseDouble("UpdateTimeout", 0.03, config);
    int updateWaitInterval = parseInt("UpdateWaitInterval", 30000, config);
    int renderWaitInterval = parseInt("RenderWaitInterval", 30000, config);

    inQueMaxSize = parseInt("InQueMaxSize", 3, config);
    renderQueMaxSize = parseInt("RenderQueMaxSize", 3, config);
    tagQueMaxSize = parseInt("TagQueMaxSize", 10, config);

    Timer tmr;

    std::string windowName = "Tagging Window";
    std::string lbfile = config["LabelFile"];
    loadLabel(lbfile);

    std::string vin = config["InputVideo"];
    cv::VideoCapture cap = cv::VideoCapture(vin);

    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::setMouseCallback(windowName, onMouse, 0);

    std::thread updateThread(updateTask, updateTimeout, updateWaitInterval);
    std::thread renderThread(renderTask, renderTimeout, renderWaitInterval);
    std::thread dumpThread(dumpTask, filename, 100, 500000);

    cv::Mat rawFrame;
    cap >> rawFrame;

    bool doMainThread = true;
    unsigned int frameId = 0;
    while (doMainThread) {
        tmr.reset();

        if (isPlay) {
            cap >> rawFrame;
            frameId++;
        }
        
        if (!rawFrame.empty() && inQue.size() < inQueMaxSize) {
            inQue.push(Frame(frameId, rawFrame));
        }
        
        while (tmr.elapsed() < mainThreadInterval) {
            doMainThread &= keyHandler(1);
        }

        if (!renderQue.empty()) {
            cv::imshow(windowName, renderQue.front());
            renderQue.pop();
        }

    }

    doUpdate = false;
    doRender = false;
    doDump = false;

    updateThread.join();
    renderThread.join();
    dumpThread.join();

    cv::destroyAllWindows();

    return 0;
}

bool parseBoolean(std::string key, bool defaultVal, e2config::Map &cfg) {
    bool val = defaultVal;
    if (cfg.iskey(key)) {
        try {
            val = std::stod(cfg[key]);
        } catch(...) {
            std::cout << "[WARNING] Parse " << key << " failed, use default: " << defaultVal << " instead.\n";
        }
    }

    return val;
}



double parseDouble(std::string key, double defaultVal, e2config::Map &cfg) {
    double val = defaultVal;
    if (cfg.iskey(key)) {
        try {
            val = std::stod(cfg[key]);
        } catch(...) {
            std::cout << "[WARNING] Parse " << key << " failed, use default: " << defaultVal << " instead.\n";
        }
    }

    return val;
}

int parseInt(std::string key, int defaultVal, e2config::Map &cfg) {
    int val = defaultVal;
    if (cfg.iskey(key)) {
        try {
            val = std::stod(cfg[key]);
        } catch(...) {
            std::cout << "[WARNING] Parse " << key << " failed, use default: " << defaultVal << " instead.\n";
        }
    }

    return val;
}

bool precheck(e2config::Map &cfg) {
    if(!cfg.iskey("InputVideo") || !cfg.iskey("LabelFile") || !cfg.iskey("OutputDir")) {
        std::cout << "Config for IO not found, please check InputVideo/LabelFile/OutputDir settings.\n";
        return false;
    }

    cv::VideoCapture cap = cv::VideoCapture(cfg["InputVideo"]);
    if(!cap.isOpened()) {
        std::cout << "[ERROR] VideoCapture: [" << cfg["InputVideo"] << "]  open failed. \n";
        return false;
    }

    std::ifstream lbfile(cfg["LabelFile"]);
    if(!lbfile.good()) {
        std::cout << "[ERROR] Open LabelFile: [" << cfg["LabelFile"] << "] failed. \n";
        return false;
    }

    if(!boost::filesystem::is_directory(cfg["OutputDir"])) {
        std::cout << "[ERROR] Output directory: ["<< cfg["OutputDir"] << "] not found. \n";
        return false;
    }

    return true;
}

void loadLabel(std::string filepath) {
    std::ifstream f(filepath);
    while(f) {
        std::string lb;
        std::getline (f, lb);
        boost::trim(lb);
        if (lb.length() > 0) labels.push_back(lb);
        
    }
    if (labels.size() > 1) labels.erase(labels.begin());
}

void loadConfig(std::string filepath, e2config::Map &cfg) {
    std::ifstream f(filepath);
    f >> cfg;
    f.close();
}


void dumpTask(std::string f, unsigned int t, unsigned int d) {
    static std::string filename = f;
    static unsigned int trunksize = t;
    static unsigned int delay = d;

    while(doDump) {
        if (outQue.size() < trunksize) {
            usleep(delay);
            continue;
        }
        std::ofstream out(filename, std::ios_base::app);
        for (int i = 0; i < trunksize; i++) {
            Tag buffer = outQue.front();
            outQue.pop();
            out << buffer.label << "\t" << buffer.uuid << "\t" << buffer.rect.x << "\t"
                << buffer.rect.y << "\t" << buffer.rect.width << "\t" << buffer.rect.height << "\n";
        }
        out.close();
    }
}

void updateTask(double t, unsigned int d) {
    static unsigned int delay = d;
    static double timeout = t;
    static int idx = 0;
    static Timer tmr;
    bool searchGrad = parseBoolean("SearchGrad", false, config);
    int erodeSize = parseInt("ErodeSize", -1, config);
    int blurSize = parseInt("BlurSize", -1, config);

    while(doUpdate) {
        tmr.reset();
        if (vecTags.empty()) {
            usleep(delay);
            continue;
        }

        while (tagQue.size() < tagQueMaxSize && outQue.size() < 100 && tmr.elapsed() < timeout) {
            if (idx >= vecTags.size() || inQue.size() <= 1) {
                idx = 0;
                while (tmr.elapsed() < timeout);
                break;
            } else {
                if (vecTags[idx].rect.area() <= 10) {
                    vecTags.erase(vecTags.begin() + idx);
                } else {
                    if (isPlay) {
                        Frame buffer = inQue.back();
                        cv::Mat targetFrame;
                        buffer.frame.copyTo(targetFrame);
                        cv::Rect local, rect;
                        search(targetFrame(vecTags[idx].srchRect), vecTags[idx].tmpl,
                            vecTags[idx].whRatio, local, blurSize, searchGrad, erodeSize 
                        );
                        rect = cv::Rect(
                            local.x + vecTags[idx].srchRect.x,
                            local.y + vecTags[idx].srchRect.y,
                            local.width, local.height
                        );
                        if (vecTags[idx].updateTemplate(buffer.frameId, targetFrame, rect)) {
                            tagQue.push(vecTags[idx]);
                            outQue.push(vecTags[idx]);
                        } else {
                            vecTags.erase(vecTags.begin() + idx);
                        }
                    } else {
                        tagQue.push(vecTags[idx]);
                    }
                    
                }

                idx++;
            }
        }

    }
    
}

void displayDebugMsg(cv::Mat & src) {
    ;
}

void displayLabel(cv::Mat & src) {
    static int fontface = CV_FONT_HERSHEY_COMPLEX;
    static double scale = 1.0;
    static int thickness = 2;
    static int baseline = 0;
    static int ofstY = 20;

    if (--displayCount == 0) {
        displayCount = displayTimeout;
        showLabel = false;
    } else {
        unsigned int num = labels.size();
        if (num > 0) {
            idxSelectedLabel = idxSelectedLabel < num ? idxSelectedLabel : num - 1;
            cv::Size textSize = cv::getTextSize(labels[idxSelectedLabel], fontface, scale, thickness, &baseline);
            cv::Point textOrg = cv::Point(0, ofstY + textSize.height - baseline);
            cv::rectangle(src, textOrg + cv::Point(0, baseline),
                cv::Point(textSize.width, -textSize.height),
                CV_RGB(0, 0, 0), CV_FILLED
            );

            cv::putText(src, labels[idxSelectedLabel], textOrg, fontface, scale, CV_RGB(150, 150, 150), thickness);
        }
    }
}

void displayRect(cv::Mat & src, const cv::Rect & r, cv::Scalar color, int thickness) {
    if (r.width > 0 && r.height > 0) {
        cv::rectangle(src, r, color, thickness);
    }
}

void displayTag(cv::Mat & src, const Tag & tag) {
    // TODO: default setting
    static int fontface = CV_FONT_HERSHEY_COMPLEX;
    static double scale = 0.8;
    static int thickness = 2;
    static int baseline = 0;

    cv::Size textSize = cv::getTextSize(tag.label, fontface, scale, thickness, &baseline);
    baseline += thickness;

    cv::Point textOrg = cv::Point(tag.rect.x - thickness + 1, tag.rect.y - baseline);

    cv::rectangle(src, textOrg + cv::Point(0, baseline),
        textOrg + cv::Point(textSize.width, -textSize.height),
        tag.color, CV_FILLED);

    cv::putText(src, tag.label, textOrg, fontface, scale, CV_RGB(255, 255, 255), thickness);
    displayRect(src, tag.rect, tag.color, thickness);

}

void renderTask(double t, unsigned int d) {
    static double timeout = t;
    static unsigned int delay = d;
    static Timer tmr;
    cv::Mat targetFrame;
    while(doRender) {
        tmr.reset();
        if (inQue.size() < 3) {
            usleep(delay);
            continue;
        }
        inQue.front().frame.copyTo(targetFrame);
        inQue.pop();

        if(showDebug) displayDebugMsg(targetFrame);
        if(showLabel) displayLabel(targetFrame);
        if(inSession) displayRect(targetFrame, rawRect);
        if(inTagMode && vecTags.size() > 0) {
            static int padding = 3;
            idxSelectedTag = idxSelectedTag < vecTags.size() ? idxSelectedTag : 0;
            int x = std::max(0, vecTags[idxSelectedTag].rect.x - padding);
            int y = std::max(0, vecTags[idxSelectedTag].rect.y - padding);
            int w = vecTags[idxSelectedTag].rect.width + 2 * padding;
            int h = vecTags[idxSelectedTag].rect.height + 2 * padding;
            cv::Rect border = cv::Rect(x, y, w, h);
            displayRect(targetFrame, border, CV_RGB(200, 200, 0), 3);
        }

        while(!tagQue.empty() && tmr.elapsed() < timeout) {
            displayTag(targetFrame, tagQue.front());
            tagQue.pop();
        }

        if (renderQue.size() < renderQueMaxSize) {
            renderQue.push(targetFrame);
        }
        
    }
}



bool keyHandler(const unsigned int delay) {
    int key;
    bool ret = true;

    // ASSERT: "false" in file qasciikey.cpp, line 501
    key = cv::waitKey(delay);
    
    if (key > 0 && key < 255) {
        unsigned int size = vecTags.size();
        unsigned int lbSize = labels.size(); 
        switch(key) {
            case 'q':
            case 27:
                ret = false;
                break;
            case 'p':
                rawRect = cv::Rect();
                inTagMode = isPlay;
                isPlay = !isPlay;
                showLabel = inTagMode;
                displayCount = displayTimeout;
                break;
            case 'x':
                if (inTagMode && idxSelectedTag < size) vecTags.erase(vecTags.begin() + idxSelectedTag);
                break;
            case ',':
                if (inTagMode && size > 0) idxSelectedTag = (--idxSelectedTag + size) % size;
                break;
            case '.':
                if (inTagMode && size > 0) idxSelectedTag = (++idxSelectedTag) % size;
                break;
            case 'w':
                if (inTagMode && idxSelectedTag < size) {
                    cv::Mat targetFrame;
                    Frame buffer;
                    buffer = inQue.front();
                    cv::Rect rect;
                    buffer.frame.copyTo(targetFrame);
                    rect = cv::Rect(vecTags[idxSelectedTag].rect.x,
                                    std::max(vecTags[idxSelectedTag].rect.y - 2, 0),
                                    vecTags[idxSelectedTag].rect.width,
                                    vecTags[idxSelectedTag].rect.height
                    );
                    vecTags[idxSelectedTag].updateTemplate(buffer.frameId, targetFrame, rect);
                }
                break;
            case 's':
                if (inTagMode && idxSelectedTag < size) {
                    cv::Mat targetFrame;
                    Frame buffer;
                    buffer = inQue.front();
                    cv::Rect rect;
                    buffer.frame.copyTo(targetFrame);
                    rect = cv::Rect(vecTags[idxSelectedTag].rect.x,
                                    std::min(vecTags[idxSelectedTag].rect.y + 2, targetFrame.rows - vecTags[idxSelectedTag].rect.height),
                                    vecTags[idxSelectedTag].rect.width,
                                    vecTags[idxSelectedTag].rect.height
                    );
                    vecTags[idxSelectedTag].updateTemplate(buffer.frameId, targetFrame, rect);
                }
                break;
            case 'a':
                if (inTagMode && idxSelectedTag < size) {
                    cv::Mat targetFrame;
                    Frame buffer;
                    buffer = inQue.front();
                    cv::Rect rect;
                    buffer.frame.copyTo(targetFrame);
                    rect = cv::Rect(std::max(vecTags[idxSelectedTag].rect.x - 2, 0),
                                    vecTags[idxSelectedTag].rect.y,
                                    vecTags[idxSelectedTag].rect.width,
                                    vecTags[idxSelectedTag].rect.height
                    );
                    vecTags[idxSelectedTag].updateTemplate(buffer.frameId, targetFrame, rect);
                }
                break;
            case 'd':
                if (inTagMode && idxSelectedTag < size) {
                    cv::Mat targetFrame;
                    Frame buffer;
                    buffer = inQue.front();
                    cv::Rect rect;
                    buffer.frame.copyTo(targetFrame);
                    rect = cv::Rect(std::min(vecTags[idxSelectedTag].rect.x + 2, targetFrame.cols - vecTags[idxSelectedTag].rect.width),
                                    vecTags[idxSelectedTag].rect.y,
                                    vecTags[idxSelectedTag].rect.width,
                                    vecTags[idxSelectedTag].rect.height
                    );
                    vecTags[idxSelectedTag].updateTemplate(buffer.frameId, targetFrame, rect);
                }
                break;
            case '+':
                if (inTagMode && idxSelectedTag < size) {
                    cv::Mat targetFrame;
                    Frame buffer;
                    buffer = inQue.front();
                    cv::Rect rect;
                    buffer.frame.copyTo(targetFrame);
                    rect = cv::Rect(
                        vecTags[idxSelectedTag].rect.x,
                        vecTags[idxSelectedTag].rect.y,
                        int(std::round(double(vecTags[idxSelectedTag].rect.width) * 1.1)),
                        int(std::round(double(vecTags[idxSelectedTag].rect.height) * 1.1))
                    );
                    vecTags[idxSelectedTag].updateTemplate(buffer.frameId, targetFrame, rect);
                }
                break;
            case '-':
                if (inTagMode && idxSelectedTag < size) {
                    cv::Mat targetFrame;
                    Frame buffer;
                    buffer = inQue.front();
                    cv::Rect rect;
                    buffer.frame.copyTo(targetFrame);
                    rect = cv::Rect(
                        vecTags[idxSelectedTag].rect.x,
                        vecTags[idxSelectedTag].rect.y,
                        int(std::round(double(vecTags[idxSelectedTag].rect.width) * 0.9)),
                        int(std::round(double(vecTags[idxSelectedTag].rect.height) * 0.9))
                    );
                    vecTags[idxSelectedTag].updateTemplate(buffer.frameId, targetFrame, rect);
                }
                break;
            case '[':
                if (lbSize > 0) idxSelectedLabel = (++idxSelectedLabel) % lbSize;
                showLabel = true;
                displayCount = displayTimeout;
                break;
            case ']':
                if (lbSize > 0) idxSelectedLabel = (--idxSelectedLabel + lbSize) % lbSize;
                showLabel = true;
                displayCount = displayTimeout;
                break;
            case 'z':
                std::cout << "inQue: " << inQue.size() << "\n";
                std::cout << "renderQue: " << renderQue.size() << "\n";
                std::cout << "tagQue: " << tagQue.size() << "\n";
                std::cout << "outQue: " << outQue.size() << "\n"; 
                std::cout << "tags: " << vecTags.size() << "\n";
                std::cout << "labels: " << labels.size() << "\n";
                break;
            default:
                std::cout << "Unkonwn key: " << key << "\n";
                ret = true;
        }
    }

    std::cout << std::flush;
    
    return ret;
}


static void onMouse(int event, int x, int y, int flag, void* ) {
    static int w, h;
    if (inTagMode) {
        if (!inSession) {
            if (event == CV_EVENT_LBUTTONUP) {
                rawRect = cv::Rect(x, y, 0, 0);
                inSession = true;
            }

        } else {

            if (event == CV_EVENT_LBUTTONUP && rawRect.area() > 0) {
                std::string lb = idxSelectedLabel < labels.size() ? labels[idxSelectedLabel] : "Unkown";
                Frame buffer = inQue.front();
                cv::Mat targetFrame;
                buffer.frame.copyTo(targetFrame);
                vecTags.push_back(Tag(targetFrame, buffer.frameId, lb, rawRect));
                inSession = false;
            }
            else if (event == CV_EVENT_MOUSEMOVE) {
                w = std::abs(rawRect.x - x);
                h = std::abs(rawRect.y - y);
                rawRect.x = x > rawRect.x ? rawRect.x : x;
                rawRect.y = y > rawRect.y ? rawRect.y : y;
                rawRect.width = w;
                rawRect.height = h;

            }
        }
    }
}


double search(const cv::Mat & srchBox, const cv::Mat & tmpl, 
    const double whRatio, cv::Rect & uRect,
    int blurSize, bool searchGrad, int erodeSize,
    double lower, double upper, double step,
    cv::Point * where, double * scale) {

    int count = int((upper - lower) / step);
    double minVal, min = 1e9;
    double factor, scl = 1.0;
    cv::Point minLoc, wh;

    cv::Mat tSrchBox, tTmpl;

    srchBox.copyTo(tSrchBox);
    tmpl.copyTo(tTmpl);

    if (blurSize > 0) {
        toGaussionBlur(tSrchBox, tSrchBox, blurSize);
        toGaussionBlur(tTmpl, tTmpl, blurSize);
    }

    if (searchGrad) {
        toXYGradient(tSrchBox, tSrchBox);
        toXYGradient(tTmpl, tTmpl);

        if (erodeSize > 0) {
            toErode(tSrchBox, tSrchBox, erodeSize);
            toErode(tTmpl, tTmpl, erodeSize);
        }
            
    }

    cv::Mat final;

    for (int i = 0; i < count; i++) {
        cv::Mat dynTmpl;
    
        factor = double(lower) + double(i) * step;
        cv::resize(tTmpl, dynTmpl, cv::Size(), factor, factor);
    
        cv::Mat result = cv::Mat(tSrchBox.rows - dynTmpl.rows + 1, tSrchBox.cols - dynTmpl.cols + 1, CV_32FC1);
        cv::matchTemplate(tSrchBox, dynTmpl, result, CV_TM_SQDIFF);
    
        cv::minMaxLoc(result, &minVal, NULL, &minLoc, NULL);
    
        if (minVal < min) {
            min = minVal;
            scl = factor;
            wh = minLoc;
            final = result;
        }
    }

    double uw, uh;
    if (tmpl.cols > tmpl.rows) {
        uw = double(tmpl.cols) * scl;
        uh = uw / whRatio;
    } else {
        uh = double(tmpl.rows) * scl;
        uw = whRatio * uh;
        
    }
    uRect = cv::Rect(wh.x, wh.y,
        (int)std::round(uw),
        (int)std::round(uh)
    );

    if (scale) *scale = scl;
    if (where) {
        where->x = wh.x;
        where->y = wh.y;
    }

    return min;

}


void toErode(cv::Mat &src, cv::Mat &dst, int ksize) {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
    cv::erode(src, dst, kernel);
}

void toGaussionBlur(cv::Mat &src, cv::Mat &dst, int ksize) {
    cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), 1.0);
}

void toXYGradient(const cv::Mat &src, cv::Mat &dsrc) {
    cv::Mat src_, temp, sobel, dx, dy, gradFg;
    if (src.channels() == 3) {
        cv::cvtColor(src, src_, CV_RGB2GRAY);
    } else {
        src.copyTo(src_);
    }
    
    src_.convertTo(temp, CV_16S);
    cv::Sobel(temp, sobel, CV_16S, 1, 0, CV_SCHARR);
    dx = cv::abs(sobel);
    cv::Sobel(temp, sobel, CV_16S, 0, 1, CV_SCHARR);
    dy = cv::abs(sobel);
    cv::addWeighted(dx, 0.5, dy, 0.5, 0, temp);
    temp.convertTo(dsrc, CV_8UC1);
}
