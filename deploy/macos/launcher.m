#import <Cocoa/Cocoa.h>
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

@interface SplashApp : NSObject <NSApplicationDelegate>
@property(nonatomic) pid_t child;
@property(nonatomic) int parentPipe;
@property(nonatomic) BOOL quitting;
@end

@implementation SplashApp

- (instancetype)init {
    self = [super init];
    if (self) {
        _child = -1;
        _parentPipe = -1;
    }
    return self;
}

- (NSString *)studioURL {
    NSString *path = [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/Splash/port"];
    NSError *error = nil;
    NSString *port = [[NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error]
                      stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSCharacterSet *digits = [NSCharacterSet decimalDigitCharacterSet];
    if (port.length == 0 || port.length > 5 || [[port stringByTrimmingCharactersInSet:digits] length] != 0) {
        port = @"8000";
    }
    return [NSString stringWithFormat:@"http://127.0.0.1:%@", port];
}

- (void)openStudio:(id)sender {
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:[self studioURL]]];
}

- (void)fail:(NSString *)message {
    self.quitting = YES;
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Splash";
    alert.informativeText = message;
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
    [NSApp terminate:nil];
}

- (void)startBackend {
    NSString *resources = [[NSBundle mainBundle] resourcePath];
    NSString *python = [resources stringByAppendingPathComponent:@"python/bin/python3"];
    NSString *script = [resources stringByAppendingPathComponent:@"launch_splash.py"];
    if (![[NSFileManager defaultManager] isExecutableFileAtPath:python]) {
        [self fail:[NSString stringWithFormat:@"The bundled Python runtime is missing at %@.", python]];
        return;
    }
    int fds[2];
    if (pipe(fds) != 0) {
        [self fail:@"Could not create the shutdown channel."];
        return;
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, fds[1]);
    NSString *fdValue = [NSString stringWithFormat:@"%d", fds[0]];
    setenv("SPLASH_PARENT_FD", [fdValue UTF8String], 1);

    const char *pythonPath = [python fileSystemRepresentation];
    const char *scriptPath = [script fileSystemRepresentation];
    char *argv[] = {(char *)pythonPath, (char *)scriptPath, NULL};
    pid_t pid = -1;
    int rc = posix_spawn(&pid, pythonPath, &actions, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(fds[0]);
    if (rc != 0) {
        close(fds[1]);
        [self fail:[NSString stringWithFormat:@"Could not start Splash (error %d).", rc]];
        return;
    }
    self.child = pid;
    self.parentPipe = fds[1];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        int status = 0;
        waitpid(pid, &status, 0);
        int code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
        dispatch_async(dispatch_get_main_queue(), ^{
            self.child = -1;
            if (!self.quitting && code != 0) {
                [self fail:@"Splash stopped. The log is ~/Library/Logs/Splash/splash.log."];
                return;
            }
            [NSApp terminate:nil];
        });
    });
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    [self startBackend];
}

- (BOOL)applicationShouldHandleReopen:(NSApplication *)sender hasVisibleWindows:(BOOL)flag {
    [self openStudio:sender];
    return NO;
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
    self.quitting = YES;
    if (self.parentPipe >= 0) {
        close(self.parentPipe);
        self.parentPipe = -1;
    }
    if (self.child > 0) {
        kill(self.child, SIGTERM);
    }
    return NSTerminateNow;
}

@end

static void installMenu(SplashApp *delegate) {
    NSMenu *menubar = [[NSMenu alloc] init];
    NSMenuItem *appItem = [[NSMenuItem alloc] init];
    [menubar addItem:appItem];
    NSMenu *appMenu = [[NSMenu alloc] initWithTitle:@"Splash"];
    NSMenuItem *openItem = [[NSMenuItem alloc] initWithTitle:@"Open Studio"
                                                      action:@selector(openStudio:)
                                               keyEquivalent:@"o"];
    openItem.target = delegate;
    [appMenu addItem:openItem];
    [appMenu addItem:[NSMenuItem separatorItem]];
    NSMenuItem *quitItem = [[NSMenuItem alloc] initWithTitle:@"Quit Splash"
                                                      action:@selector(terminate:)
                                               keyEquivalent:@"q"];
    quitItem.target = NSApp;
    [appMenu addItem:quitItem];
    appItem.submenu = appMenu;
    NSApp.mainMenu = menubar;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        SplashApp *delegate = [[SplashApp alloc] init];
        app.delegate = delegate;
        installMenu(delegate);
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        [app run];
    }
    return 0;
}
